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

vehicle_model  = YOLO("models_and_weights/vehicle_elite.pt")   # Rezuwan: vehicles+pedestrians+signs
traffic_model  = YOLO("models_and_weights/traffic_signs.pt")   # ablanco: traffic signs/lights

print("[INFO] Loading lane segmentation model...")
lane_processor = SegformerImageProcessor.from_pretrained("bricklerex/lane-detect-jds")
lane_model     = SegformerForSemanticSegmentation.from_pretrained("bricklerex/lane-detect-jds")
lane_model.eval()

print("[INFO] All models loaded!")

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
font      = cv2.FONT_HERSHEY_SIMPLEX
FOCAL_LEN = 250.0   # pixels (camera calibration approx)

# Class heights in meters (for TTC distance estimation, ravesandstorm style)
CLASS_HEIGHTS = {
    "car": 1.5, "vehicle": 1.5, "bus": 3.0, "truck": 3.5,
    "motorcycle": 1.2, "person": 1.7, "pedestrian": 1.7,
    "bicycle": 1.1, "signboard": 2.0,
}

# ─────────────────────────────────────────────────────────────────
#  WARNING STATE GLOBALS (Ashish-style)
# ─────────────────────────────────────────────────────────────────
crash_count_frames = 0
flagPerson         = 0
areaPerson         = 0
areaDetails        = []
signalCounter      = -99999
flagSignal         = [0] * 20
flagLanes          = [0] * 20
prev_frame         = []
number             = 0

# ─────────────────────────────────────────────────────────────────
#  TTC CALCULATION (ravesandstorm-style)
# ─────────────────────────────────────────────────────────────────
speed_buffer = []

def estimate_distance(bbox_h_pixels, class_name):
    """Estimate distance using pinhole camera model."""
    real_h = CLASS_HEIGHTS.get(class_name.lower(), 1.5)
    if bbox_h_pixels < 1:
        return 999.0
    return (real_h * FOCAL_LEN) / bbox_h_pixels

def get_smoothed_speed(new_speed, alpha=0.3):
    """Exponential moving average of speed (km/h)."""
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
    """TTC = distance / speed (ravesandstorm formula)."""
    speed_ms = speed_kmh / 3.6
    if speed_ms < 0.5:
        return 999.0
    return distance_m / speed_ms

# ─────────────────────────────────────────────────────────────────
#  LANE SEGMENTATION (bricklerex/lane-detect-jds)
# ─────────────────────────────────────────────────────────────────
def run_lane_segmentation(frame):
    """Returns a lane overlay mask on the frame."""
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    inputs = lane_processor(images=pil, return_tensors="pt")
    with torch.no_grad():
        outputs = lane_model(**inputs)
    logits    = outputs.logits                          # (1, num_labels, H, W)
    upsampled = F.interpolate(logits, size=frame.shape[:2],
                              mode="bilinear", align_corners=False)
    seg_mask  = upsampled.argmax(dim=1).squeeze().numpy().astype(np.uint8)

    overlay = frame.copy()
    lane_px = seg_mask == 1                             # class 1 = lane
    overlay[lane_px] = (overlay[lane_px] * 0.5 + np.array([0, 255, 100]) * 0.5).astype(np.uint8)
    return overlay, lane_px

# ─────────────────────────────────────────────────────────────────
#  SIGNAL DETECTION (Ashish-style HSV red detection)
# ─────────────────────────────────────────────────────────────────
startRedLower = (0,   130, 50)
startRedUpper = (13,  255, 255)
endRedLower   = (150, 130, 50)
endRedUpper   = (180, 255, 255)

def detect_red_signal(image_np, light_boxes, signalCounter, flagSignal):
    maskRed = np.zeros_like(image_np)
    for (x, y, w, h) in light_boxes:
        if w * h > 450:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (255, 255, 0), 2)
        maskRed[y:y+h, x:x+w] = image_np[y:y+h, x:x+w]

    blurred = cv2.GaussianBlur(maskRed, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    m1      = cv2.inRange(hsv, startRedLower, startRedUpper)
    m2      = cv2.inRange(hsv, endRedLower,   endRedUpper)
    red     = cv2.erode(m1 + m2, None, 2)
    red     = cv2.dilate(red, None, 2)

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = 0
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        top  = tuple(hull[hull[:, :, 1].argmin()][0])
        bot  = tuple(hull[hull[:, :, 1].argmax()][0])
        lft  = tuple(hull[hull[:, :, 0].argmin()][0])
        rgt  = tuple(hull[hull[:, :, 0].argmax()][0])
        cX   = (lft[0] + rgt[0]) // 2
        cY   = (top[1] + bot[1]) // 2
        dists = pairwise.euclidean_distances([(cX, cY)],
                                             Y=[lft, rgt, top, bot])[0]
        if int(dists.max()) >= 3:
            flag = 1

    flagSignal.pop(0)
    flagSignal.append(flag)

    if sum(flagSignal) > 5:
        cv2.putText(image_np, "!! TRAFFIC SIGNAL IS RED !!", (300, 160),
                    font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        signalCounter = 1
    else:
        signalCounter -= 1
    if -16 < signalCounter <= 0:
        cv2.putText(image_np, "You can drive now", (380, 160),
                    font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

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
            areaPerson    = curr_area
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
            cv2.putText(image_np, "STOP! DON'T HIT THE PERSON", (260, 120),
                        font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_np, "Drive slowly, people around", (280, 120),
                        font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    return image_np, flagPerson, areaPerson, areaDetails

# ─────────────────────────────────────────────────────────────────
#  VEHICLE COLLISION + TTC WARNING
# ─────────────────────────────────────────────────────────────────
def estimate_collide_ttc(car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh=40.0):
    height, width, _ = image_np.shape
    max_area  = 0
    best_box  = None
    best_cls  = "car"

    for (x, y, w, h), cls in zip(car_boxes, car_classes):
        obj_area = w * h
        if obj_area > max_area:
            max_area = obj_area
            best_box = (x, y, w, h)
            best_cls = cls

    if best_box is None:
        crash_count_frames = max(0, crash_count_frames - 1)
        return image_np, crash_count_frames

    x, y, w, h   = best_box
    centerX      = (x + w / 2) / width
    centerY      = (y + h / 2) / height

    # TTC calculation (ravesandstorm formula)
    distance_m   = estimate_distance(h, best_cls)
    smoothed_spd = get_smoothed_speed(ego_speed_kmh)
    ttc          = compute_ttc(distance_m, smoothed_spd)

    # Ashish-style area thresholds
    if max_area > 40000 and 0.27 <= centerX <= 0.73:
        crash_count_frames = 10

    if crash_count_frames > 0:
        # TTC-based severity (ravesandstorm threshold: 0.65s)
        if ttc < 0.65:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(image_np, f"!!! COLLISION IMMINENT !!! TTC:{ttc:.1f}s",
                        (200, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        elif ttc < 2.0:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 165, 255), 3)
            cv2.putText(image_np, f"DANGER - TTC: {ttc:.1f}s  Dist: {distance_m:.1f}m",
                        (280, 40), font, 1.2, (0, 165, 255), 2, cv2.LINE_AA)
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

    # HUD: distance + speed
    cv2.putText(image_np, f"Dist: {distance_m:.1f}m | Speed: {smoothed_spd:.0f}km/h | TTC: {ttc:.1f}s",
                (10, height - 15), font, 0.7, (200, 200, 200), 2)

    return image_np, crash_count_frames

# ─────────────────────────────────────────────────────────────────
#  MULTIMODAL INFERENCE (main frame processor)
# ─────────────────────────────────────────────────────────────────
def run_multimodal(frame, ego_speed_kmh=40.0, run_lanes=True):
    global crash_count_frames, flagPerson, areaPerson, areaDetails
    global signalCounter, flagSignal, flagLanes, prev_frame, number

    image_np = np.array(frame)
    height, width, _ = image_np.shape

    # ── 1. LANE SEGMENTATION ─────────────────────────────────────
    if run_lanes:
        image_np, _ = run_lane_segmentation(image_np)

    # ── 2. VEHICLE DETECTION (vehicle_elite.pt) ──────────────────
    veh_res    = vehicle_model(frame, verbose=False, conf=0.3)
    car_boxes, car_classes   = [], []
    person_boxes             = []

    for box in veh_res[0].boxes:
        cls_name = vehicle_model.names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        conf = float(box.conf)

        if any(k in cls_name for k in ["car", "vehicle", "bus", "truck", "motorcycle", "bicycle"]):
            car_boxes.append((x, y, w, h))
            car_classes.append(cls_name)
            color = (255, 180, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        font, 0.6, color, 2)
        elif any(k in cls_name for k in ["person", "pedestrian"]):
            person_boxes.append((x, y, w, h))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (50, 50, 255), 2)
            cv2.putText(image_np, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        font, 0.6, (50, 50, 255), 2)

    # ── 3. TRAFFIC SIGN/LIGHT DETECTION ──────────────────────────
    sign_res   = traffic_model(frame, verbose=False, conf=0.4)
    light_boxes = []

    for box in sign_res[0].boxes:
        cls_name = traffic_model.names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        conf = float(box.conf)
        light_boxes.append((x, y, w, h))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image_np, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                    font, 0.6, (0, 255, 255), 2)

    # ── 4. WARNING LOGIC (Ashish-style) ──────────────────────────
    image_np, crash_count_frames = estimate_collide_ttc(
        car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh)

    image_np, flagPerson, areaPerson, areaDetails = estimate_stepping(
        person_boxes, image_np, flagPerson, areaPerson, areaDetails)

    image_np, signalCounter, flagSignal = detect_red_signal(
        image_np, light_boxes, signalCounter, flagSignal)

    # ── 5. HUD overlay ───────────────────────────────────────────
    cv2.putText(image_np, "MULTIMODAL ADAS", (10, 25),
                font, 0.8, (0, 255, 0), 2)
    cv2.putText(image_np, f"Vehicles:{len(car_boxes)} | People:{len(person_boxes)} | Signs:{len(light_boxes)}",
                (10, 55), font, 0.65, (200, 200, 200), 1)

    return image_np


# ─────────────────────────────────────────────────────────────────
#  ROI SELECTION (Ashish-style manual click)
# ─────────────────────────────────────────────────────────────────
refPt = []

def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])

def confirm_day_or_night(frame, flag_night_counter):
    blurred   = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv       = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask      = cv2.inRange(hsv, (0, 0, 0), (180, 255, 35))
    mask      = cv2.dilate(cv2.erode(mask, None, 2), None, 2)
    ratio     = np.sum(mask == 0) / mask.size
    return flag_night_counter + (1 if ratio < 0.5 else -1)


# ─────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="0",        help="Video path or 0 for webcam")
    parser.add_argument("--speed",  type=float, default=40.0, help="Ego speed km/h (or use GPS)")
    parser.add_argument("--no-lanes", action="store_true",   help="Disable lane segmentation (faster)")
    args = parser.parse_args()

    src = int(args.video) if args.video == "0" else args.video
    cap = cv2.VideoCapture(src)

    # Day/Night check
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
        frame = imutils.resize(frame, width=1280)

        result = run_multimodal(
            frame,
            ego_speed_kmh=args.speed,
            run_lanes=(not args.no_lanes and not is_night)
        )

        fps.update()
        elapsed = fps.elapsed()
        curr_fps = fps._numFrames / elapsed if elapsed > 0 else 0
        cv2.putText(result, f"FPS: {curr_fps:.1f}", (1150, 25),
                    font, 0.7, (0, 255, 0), 2)

        cv2.imshow("MULTIMODAL ADAS", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print(f"[INFO] Elapsed: {fps.elapsed():.2f}s | FPS: {fps.fps():.2f}")
    cap.release()
    cv2.destroyAllWindows()
