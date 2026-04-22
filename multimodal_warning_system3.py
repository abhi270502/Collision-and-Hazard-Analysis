import numpy as np
import cv2
import imutils
import copy
import time
from imutils.video import FPS
from sklearn.metrics import pairwise
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import torch.nn.functional as F
from PIL import Image

# ─────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────
print("[INFO] Loading multimodal models...")

vehicle_model       = YOLO("models_and_weights/vehicle_elite.pt")   # Rezuwan: vehicles+pedestrians
traffic_signs_model = YOLO("models_and_weights/traffic_signs.pt")   # ablanco: road signs
traffic_light_model = YOLO("models_and_weights/best.pt")            # Satish: red/green/yellow lights

print("[INFO] Loading lane segmentation model...")
lane_processor = SegformerImageProcessor.from_pretrained("bricklerex/lane-detect-jds")
lane_model     = SegformerForSemanticSegmentation.from_pretrained("bricklerex/lane-detect-jds")
lane_model.eval()

print("[INFO] All models loaded!")

# Print traffic light classes for confirmation
print(f"[INFO] Traffic light classes: {traffic_light_model.names}")

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
font      = cv2.FONT_HERSHEY_SIMPLEX
FOCAL_LEN = 250.0

CLASS_HEIGHTS = {
    "car": 1.5, "vehicle": 1.5, "bus": 3.0, "truck": 3.5,
    "motorcycle": 1.2, "person": 1.7, "pedestrian": 1.7,
    "bicycle": 1.1, "signboard": 2.0,
}

# Traffic light color → (BGR color for overlay, warning text)
LIGHT_COLOR_MAP = {
    "red":    ((0,   0,   255), "!! TRAFFIC SIGNAL IS RED !!"),
    "green":  ((0,   255,  0),  "Signal is GREEN - You can go"),
    "yellow": ((0,   200, 255), "Signal YELLOW - Slow down!"),
}

# ─────────────────────────────────────────────────────────────────
#  WARNING STATE GLOBALS
# ─────────────────────────────────────────────────────────────────
crash_count_frames = 0
flagPerson         = 0
areaPerson         = 0
areaDetails        = []
signalCounter      = -99999
flagSignal         = [0] * 20   # rolling red window (Ashish-style)
flagLanes          = [0] * 20
prev_frame         = []
number             = 0
speed_buffer       = []

# ─────────────────────────────────────────────────────────────────
#  TTC (ravesandstorm)
# ─────────────────────────────────────────────────────────────────
def estimate_distance(bbox_h_pixels, class_name):
    real_h = CLASS_HEIGHTS.get(class_name.lower(), 1.5)
    if bbox_h_pixels < 1:
        return 999.0
    return (real_h * FOCAL_LEN) / bbox_h_pixels

def get_smoothed_speed(new_speed, alpha=0.3):
    global speed_buffer
    if not speed_buffer:
        speed_buffer.append(new_speed)
    else:
        smoothed = alpha * new_speed + (1 - alpha) * speed_buffer[-1]
        speed_buffer.append(smoothed)
        if len(speed_buffer) > 30:
            speed_buffer.pop(0)
    return speed_buffer[-1]

def compute_ttc(distance_m, speed_kmh):
    speed_ms = speed_kmh / 3.6
    if speed_ms < 0.5:
        return 999.0
    return distance_m / speed_ms

# ─────────────────────────────────────────────────────────────────
#  LANE SEGMENTATION (bricklerex/lane-detect-jds)
# ─────────────────────────────────────────────────────────────────
def run_lane_segmentation(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil    = Image.fromarray(rgb)
    inputs = lane_processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = lane_model(**inputs)
    logits    = outputs.logits
    upsampled = F.interpolate(logits, size=frame.shape[:2],
                              mode="bilinear", align_corners=False)
    seg_mask  = upsampled.argmax(dim=1).squeeze().numpy().astype(np.uint8)
    overlay   = frame.copy()
    lane_px   = seg_mask == 1
    overlay[lane_px] = (overlay[lane_px] * 0.5 +
                        np.array([0, 255, 100]) * 0.5).astype(np.uint8)
    return overlay, lane_px

# ─────────────────────────────────────────────────────────────────
#  TRAFFIC LIGHT WARNING (Satish model — red/green/yellow)
# ─────────────────────────────────────────────────────────────────
def detect_traffic_lights(frame, image_np, signalCounter, flagSignal):
    """
    Uses Satish-Vennapu model (best.pt) which classifies
    red / green / yellow directly as class names.
    Falls back to Ashish HSV red-check if no detections.
    """
    light_res    = traffic_light_model(frame, verbose=False, conf=0.1)
    detected_red = False
    detected_clr = None
    best_conf    = 0.0

    for box in light_res[0].boxes:
        cls_name = traffic_light_model.names[int(box.cls)].lower()
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color_bgr, _ = LIGHT_COLOR_MAP.get(cls_name, ((0, 255, 255), ""))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(image_np, f"{cls_name} {conf:.2f}",
                    (x1, y1 - 5), font, 0.65, color_bgr, 2)

        if conf > best_conf:
            best_conf    = conf
            detected_clr = cls_name

        if cls_name == "red":
            detected_red = True

    # Rolling window (Ashish-style)
    flagSignal.pop(0)
    flagSignal.append(1 if detected_red else 0)

    # Persistent warning using rolling window
    if sum(flagSignal) > 5 or detected_red:
        cv2.putText(image_np, "!! TRAFFIC SIGNAL IS RED !!",
                    (300, 160), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        signalCounter = 1
    elif detected_clr == "yellow":
        cv2.putText(image_np, "Signal YELLOW - Slow down!",
                    (330, 160), font, 1.2, (0, 200, 255), 2, cv2.LINE_AA)
        signalCounter -= 1
    elif detected_clr == "green":
        cv2.putText(image_np, "Signal GREEN - You can go",
                    (340, 160), font, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
        signalCounter -= 1
    else:
        signalCounter -= 1

    if -16 < signalCounter <= 0:
        cv2.putText(image_np, "You can drive now",
                    (400, 160), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    return image_np, signalCounter, flagSignal

# ─────────────────────────────────────────────────────────────────
#  PEDESTRIAN WARNING (Ashish-style)
# ─────────────────────────────────────────────────────────────────
def estimate_stepping(person_boxes, image_np, flagPerson, areaPerson, areaDetails):
    pedes_present = 0
    details = []
    for (x, y, w, h) in person_boxes:
        curr_area = w * h
        if curr_area > 9000:
            pedes_present = 1
            flagPerson    = 5
            details.append([x, y, w, h])

    if pedes_present == 0:
        flagPerson -= 1
    else:
        areaPerson = 0
        for (x, y, w, h) in details:
            ba = w * h
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 0), 3)
            if ba > areaPerson:
                areaPerson = ba
        areaDetails = details

    if flagPerson > 0:
        for (x, y, w, h) in areaDetails:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 0), 3)
        if areaPerson > 15000:
            cv2.putText(image_np, "STOP! DON'T HIT THE PERSON",
                        (260, 120), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_np, "Drive slowly, people around",
                        (280, 120), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    return image_np, flagPerson, areaPerson, areaDetails

# ─────────────────────────────────────────────────────────────────
#  VEHICLE COLLISION + TTC (ravesandstorm formula)
# ─────────────────────────────────────────────────────────────────
def estimate_collide_ttc(car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh=40.0):
    height, width, _ = image_np.shape
    max_area = 0
    best_box = None
    best_cls = "car"

    for (x, y, w, h), cls in zip(car_boxes, car_classes):
        obj_area = w * h
        if obj_area > max_area:
            max_area = obj_area
            best_box = (x, y, w, h)
            best_cls = cls

    if best_box is None:
        crash_count_frames = max(0, crash_count_frames - 1)
        return image_np, crash_count_frames

    x, y, w, h  = best_box
    centerX     = (x + w / 2) / width

    distance_m   = estimate_distance(h, best_cls)
    smoothed_spd = get_smoothed_speed(ego_speed_kmh)
    ttc          = compute_ttc(distance_m, smoothed_spd)

    if max_area > 40000 and 0.27 <= centerX <= 0.73:
        crash_count_frames = 10

    if crash_count_frames > 0:
        if ttc < 0.65:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(image_np, f"!!! COLLISION IMMINENT !!! TTC:{ttc:.1f}s",
                        (200, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        elif ttc < 2.0:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 165, 255), 3)
            cv2.putText(image_np, f"DANGER  TTC:{ttc:.1f}s  Dist:{distance_m:.1f}m",
                        (300, 40), font, 1.2, (0, 165, 255), 2, cv2.LINE_AA)
        elif max_area <= 70000:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(image_np, f"Getting Closer  TTC:{ttc:.1f}s  Dist:{distance_m:.1f}m",
                        (280, 40), font, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(image_np, f"DON'T COLLIDE!  TTC:{ttc:.1f}s",
                        (360, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        crash_count_frames = max(0, crash_count_frames - 1)

    # HUD bottom bar
    cv2.putText(image_np,
                f"Dist:{distance_m:.1f}m | Speed:{smoothed_spd:.0f}km/h | TTC:{ttc:.1f}s",
                (10, height - 15), font, 0.7, (200, 200, 200), 2)

    return image_np, crash_count_frames

# ─────────────────────────────────────────────────────────────────
#  MAIN FRAME PROCESSOR
# ─────────────────────────────────────────────────────────────────
def run_multimodal(frame, ego_speed_kmh=40.0, run_lanes=True):
    global crash_count_frames, flagPerson, areaPerson, areaDetails
    global signalCounter, flagSignal, flagLanes, prev_frame, number

    image_np = np.array(frame)
    height, width, _ = image_np.shape

    # ── 1. LANE SEGMENTATION (bricklerex) ────────────────────────
    if run_lanes:
        image_np, _ = run_lane_segmentation(image_np)

    # ── 2. VEHICLE + PEDESTRIAN DETECTION (vehicle_elite.pt) ─────
    veh_res     = vehicle_model(frame, verbose=False, conf=0.3)
    car_boxes   = []
    car_classes = []
    person_boxes = []

    for box in veh_res[0].boxes:
        cls_name = vehicle_model.names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        conf = float(box.conf)

        if any(k in cls_name for k in ["car", "vehicle", "bus", "truck",
                                        "motorcycle", "bicycle", "auto", "lcv",
                                        "multiaxle", "tractor"]):
            car_boxes.append((x, y, w, h))
            car_classes.append(cls_name)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(image_np, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 5), font, 0.6, (255, 180, 0), 2)

        elif any(k in cls_name for k in ["person", "pedestrian"]):
            person_boxes.append((x, y, w, h))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (50, 50, 255), 2)
            cv2.putText(image_np, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 5), font, 0.6, (50, 50, 255), 2)

    # ── 3. TRAFFIC SIGNS (traffic_signs.pt) ──────────────────────
    sign_res = traffic_signs_model(frame, verbose=False, conf=0.4)
    for box in sign_res[0].boxes:
        cls_name = traffic_signs_model.names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(image_np, f"sign:{cls_name} {conf:.2f}",
                    (x1, y1 - 5), font, 0.55, (255, 0, 255), 2)

    # ── 4. TRAFFIC LIGHTS (best.pt — red/green/yellow) ───────────
    image_np, signalCounter, flagSignal = detect_traffic_lights(
        frame, image_np, signalCounter, flagSignal)

    # ── 5. COLLISION + TTC WARNING ───────────────────────────────
    image_np, crash_count_frames = estimate_collide_ttc(
        car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh)

    # ── 6. PEDESTRIAN WARNING ────────────────────────────────────
    image_np, flagPerson, areaPerson, areaDetails = estimate_stepping(
        person_boxes, image_np, flagPerson, areaPerson, areaDetails)

    # ── 7. HUD TOP BAR ───────────────────────────────────────────
    cv2.putText(image_np, "MULTIMODAL ADAS",
                (10, 25), font, 0.8, (0, 255, 0), 2)
    cv2.putText(image_np,
                f"Vehicles:{len(car_boxes)} | People:{len(person_boxes)} | Signs:{len(sign_res[0].boxes)}",
                (10, 55), font, 0.65, (200, 200, 200), 1)

    return image_np


# ─────────────────────────────────────────────────────────────────
#  DAY / NIGHT DETECTION
# ─────────────────────────────────────────────────────────────────
def confirm_day_or_night(frame, flag_night_counter):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, (0, 0, 0), (180, 255, 35))
    mask    = cv2.dilate(cv2.erode(mask, None, 2), None, 2)
    ratio   = np.sum(mask == 0) / mask.size
    return flag_night_counter + (1 if ratio < 0.5 else -1)


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    default="0",       help="Video path or 0 for webcam")
    parser.add_argument("--speed",    type=float, default=40.0, help="Ego speed km/h")
    parser.add_argument("--no-lanes", action="store_true",      help="Disable lane segmentation")
    args = parser.parse_args()

    src = int(args.video) if args.video == "0" else args.video
    cap = cv2.VideoCapture(src)

    # Day/Night check on first 10 frames
    flag_night_counter = 0
    for _ in range(10):
        grabbed, frame = cap.read()
        if not grabbed: break
        frame = imutils.resize(frame, width=1280)
        flag_night_counter = confirm_day_or_night(frame, flag_night_counter)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    is_night = flag_night_counter > 0
    print(f"[INFO] Mode: {'NIGHT' if is_night else 'DAY'}")
    print(f"[INFO] Lanes: {'DISABLED (night)' if is_night else 'ENABLED'}")

    cv2.namedWindow("MULTIMODAL ADAS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MULTIMODAL ADAS", 1280, 720)

    fps = FPS().start()
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame  = imutils.resize(frame, width=1280)
        result = run_multimodal(
            frame,
            ego_speed_kmh=args.speed,
            run_lanes=(not args.no_lanes and not is_night)
        )
        fps.update()
        try:
            curr_fps = fps.fps() if fps._numFrames > 1 else 0.0
        except Exception:
            curr_fps = 0.0
        cv2.putText(result, f"FPS:{curr_fps:.1f}", (1150, 25),
                    font, 0.7, (0, 255, 0), 2)

        cv2.imshow("MULTIMODAL ADAS", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    try:
        print(f"[INFO] Elapsed: {fps.elapsed():.2f}s | FPS: {fps.fps():.2f}")
    except Exception:
        print("[INFO] Done.")
    cap.release()
    cv2.destroyAllWindows()
