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

vehicle_model       = YOLO("models_and_weights/vehicle_elite.pt")
# Classes: {0:'pedestrians', 1:'signpost', 2:'vehicles'}

traffic_signs_model = YOLO("models_and_weights/traffic_signs.pt")
# Classes: {0:'prohibitory', 1:'danger', 2:'mandatory', 3:'other'}

traffic_light_model = YOLO("models_and_weights/best.pt")
# Classes: {0:'green', 1:'red', 2:'yellow'}

print("[INFO] Loading lane segmentation model...")
lane_processor = SegformerImageProcessor.from_pretrained("bricklerex/lane-detect-jds")
lane_model     = SegformerForSemanticSegmentation.from_pretrained("bricklerex/lane-detect-jds")
lane_model.eval()
print("[INFO] All models loaded!")

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────
font      = cv2.FONT_HERSHEY_SIMPLEX
FOCAL_LEN = 250.0

CLASS_HEIGHTS = {
    "vehicles": 1.5, "vehicle": 1.5, "car": 1.5,
    "pedestrians": 1.7, "pedestrian": 1.7, "person": 1.7,
    "signpost": 2.0,
}

# Traffic light color → (BGR, warning text)
LIGHT_COLORS = {
    "red":    ((0,   0,   255), "!! TRAFFIC SIGNAL IS RED !!"),
    "green":  ((0,   230,  0),  "Signal GREEN — You can go"),
    "yellow": ((0,   200, 255), "Signal YELLOW — Slow down!"),
}

# Sign category → (BGR, label)
SIGN_COLORS = {
    "prohibitory": ((0, 0, 200),   "PROHIBITORY SIGN"),
    "danger":      ((0, 165, 255), "DANGER SIGN"),
    "mandatory":   ((200, 100, 0), "MANDATORY SIGN"),
    "other":       ((180, 180, 0), "TRAFFIC SIGN"),
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
speed_buffer       = []

# ─────────────────────────────────────────────────────────────────
#  TTC — ravesandstorm formula
# ─────────────────────────────────────────────────────────────────
def estimate_distance(bbox_h_px, class_name):
    real_h = CLASS_HEIGHTS.get(class_name.lower(), 1.5)
    return (real_h * FOCAL_LEN) / bbox_h_px if bbox_h_px > 1 else 999.0

def get_smoothed_speed(new_speed, alpha=0.3):
    global speed_buffer
    if not speed_buffer:
        speed_buffer.append(new_speed)
    else:
        s = alpha * new_speed + (1 - alpha) * speed_buffer[-1]
        speed_buffer.append(s)
        if len(speed_buffer) > 30: speed_buffer.pop(0)
    return speed_buffer[-1]

def compute_ttc(dist_m, speed_kmh):
    spd_ms = speed_kmh / 3.6
    return dist_m / spd_ms if spd_ms > 0.5 else 999.0

# ─────────────────────────────────────────────────────────────────
#  LANE SEGMENTATION
# ─────────────────────────────────────────────────────────────────
def run_lane_segmentation(frame):
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = lane_processor(images=Image.fromarray(rgb), return_tensors="pt")
    with torch.no_grad():
        logits = lane_model(**inputs).logits
    up  = F.interpolate(logits, size=frame.shape[:2], mode="bilinear", align_corners=False)
    seg = up.argmax(dim=1).squeeze().numpy().astype(np.uint8)
    out = frame.copy()
    out[seg == 1] = (out[seg == 1] * 0.5 + np.array([0, 255, 100]) * 0.5).astype(np.uint8)
    return out

# ─────────────────────────────────────────────────────────────────
#  TRAFFIC LIGHT DETECTION (best.pt: green/red/yellow)
# ─────────────────────────────────────────────────────────────────
def detect_traffic_lights(frame, image_np, signalCounter, flagSignal):
    res = traffic_light_model(frame, verbose=False, conf=0.15)
    detected_red = False
    best_cls, best_conf = None, 0.0

    for box in res[0].boxes:
        cls_name = traffic_light_model.names[int(box.cls)].lower()
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bgr, _   = LIGHT_COLORS.get(cls_name, ((0, 255, 255), ""))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(image_np, f"{cls_name} {conf:.2f}", (x1, y1 - 5), font, 0.65, bgr, 2)
        if conf > best_conf:
            best_conf = conf
            best_cls  = cls_name
        if cls_name == "red":
            detected_red = True

    flagSignal.pop(0)
    flagSignal.append(1 if detected_red else 0)

    if sum(flagSignal) > 5 or detected_red:
        cv2.putText(image_np, "!! TRAFFIC SIGNAL IS RED !!",
                    (300, 200), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        signalCounter = 1
    elif best_cls == "yellow":
        cv2.putText(image_np, "Signal YELLOW — Slow down!",
                    (330, 200), font, 1.2, (0, 200, 255), 2, cv2.LINE_AA)
        signalCounter -= 1
    elif best_cls == "green":
        cv2.putText(image_np, "Signal GREEN — You can go",
                    (340, 200), font, 1.1, (0, 230, 0), 2, cv2.LINE_AA)
        signalCounter -= 1
    else:
        signalCounter -= 1

    if -16 < signalCounter <= 0:
        cv2.putText(image_np, "You can drive now",
                    (400, 200), font, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

    return image_np, signalCounter, flagSignal

# ─────────────────────────────────────────────────────────────────
#  TRAFFIC SIGNS (traffic_signs.pt: prohibitory/danger/mandatory/other)
# ─────────────────────────────────────────────────────────────────
def detect_traffic_signs(frame, image_np):
    res = traffic_signs_model(frame, verbose=False, conf=0.35)
    for box in res[0].boxes:
        cls_name = traffic_signs_model.names[int(box.cls)].lower()
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bgr, label = SIGN_COLORS.get(cls_name, ((180, 180, 0), "SIGN"))
        cv2.rectangle(image_np, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(image_np, f"{label} {conf:.2f}", (x1, y1 - 5), font, 0.6, bgr, 2)

        # Extra warning for prohibitory signs (stop, no entry etc.)
        if cls_name == "prohibitory":
            cv2.putText(image_np, "!! PROHIBITORY SIGN AHEAD !!",
                        (250, 240), font, 1.0, (0, 0, 200), 2, cv2.LINE_AA)
        elif cls_name == "danger":
            cv2.putText(image_np, "! DANGER SIGN AHEAD !",
                        (310, 240), font, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
    return image_np

# ─────────────────────────────────────────────────────────────────
#  PEDESTRIAN WARNING
# ─────────────────────────────────────────────────────────────────
def estimate_stepping(person_boxes, image_np, flagPerson, areaPerson, areaDetails):
    pedes_present = 0
    details = []
    for (x, y, w, h) in person_boxes:
        if w * h > 9000:
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
            if ba > areaPerson: areaPerson = ba
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
#  COLLISION + TTC WARNING
# ─────────────────────────────────────────────────────────────────
def estimate_collide_ttc(car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh):
    height, width, _ = image_np.shape
    max_area, best_box, best_cls = 0, None, "vehicles"

    for (x, y, w, h), cls in zip(car_boxes, car_classes):
        if w * h > max_area:
            max_area  = w * h
            best_box  = (x, y, w, h)
            best_cls  = cls

    if best_box is None:
        crash_count_frames = max(0, crash_count_frames - 1)
        return image_np, crash_count_frames

    x, y, w, h = best_box
    centerX    = (x + w / 2) / width
    dist_m     = estimate_distance(h, best_cls)
    spd        = get_smoothed_speed(ego_speed_kmh)
    ttc        = compute_ttc(dist_m, spd)

    if max_area > 40000 and 0.27 <= centerX <= 0.73:
        crash_count_frames = 10

    if crash_count_frames > 0:
        if ttc < 0.65:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(image_np, f"!!! COLLISION IMMINENT !!! TTC:{ttc:.1f}s",
                        (200, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        elif ttc < 2.0:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 165, 255), 3)
            cv2.putText(image_np, f"DANGER  TTC:{ttc:.1f}s  Dist:{dist_m:.1f}m",
                        (300, 40), font, 1.2, (0, 165, 255), 2, cv2.LINE_AA)
        elif max_area <= 70000:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(image_np, f"Getting Closer  TTC:{ttc:.1f}s  Dist:{dist_m:.1f}m",
                        (280, 40), font, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(image_np, f"DON'T COLLIDE!  TTC:{ttc:.1f}s",
                        (360, 40), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        crash_count_frames = max(0, crash_count_frames - 1)

    cv2.putText(image_np,
                f"Dist:{dist_m:.1f}m | Speed:{spd:.0f}km/h | TTC:{ttc:.1f}s",
                (10, height - 15), font, 0.7, (200, 200, 200), 2)

    return image_np, crash_count_frames

# ─────────────────────────────────────────────────────────────────
#  MAIN FRAME PROCESSOR
# ─────────────────────────────────────────────────────────────────
def run_multimodal(frame, ego_speed_kmh=40.0, run_lanes=True):
    global crash_count_frames, flagPerson, areaPerson, areaDetails
    global signalCounter, flagSignal

    image_np = np.array(frame)
    height, width, _ = image_np.shape

    # 1. LANE SEGMENTATION
    if run_lanes:
        image_np = run_lane_segmentation(image_np)

    # 2. VEHICLE + PEDESTRIAN DETECTION
    veh_res     = vehicle_model(frame, verbose=False, conf=0.3)
    car_boxes, car_classes, person_boxes = [], [], []

    for box in veh_res[0].boxes:
        cls_name = vehicle_model.names[int(box.cls)].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x, y, w, h = x1, y1, x2-x1, y2-y1
        conf = float(box.conf)

        if cls_name == "vehicles":
            car_boxes.append((x, y, w, h))
            car_classes.append(cls_name)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.putText(image_np, f"vehicle {conf:.2f}", (x1, y1-5), font, 0.6, (255, 180, 0), 2)
        elif cls_name == "pedestrians":
            person_boxes.append((x, y, w, h))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (50, 50, 255), 2)
            cv2.putText(image_np, f"person {conf:.2f}", (x1, y1-5), font, 0.6, (50, 50, 255), 2)
        elif cls_name == "signpost":
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (180, 255, 180), 1)
            cv2.putText(image_np, f"signpost {conf:.2f}", (x1, y1-5), font, 0.5, (180, 255, 180), 1)

    # 3. TRAFFIC SIGNS
    image_np = detect_traffic_signs(frame, image_np)

    # 4. TRAFFIC LIGHTS
    image_np, signalCounter, flagSignal = detect_traffic_lights(
        frame, image_np, signalCounter, flagSignal)

    # 5. COLLISION + TTC
    image_np, crash_count_frames = estimate_collide_ttc(
        car_boxes, car_classes, image_np, crash_count_frames, ego_speed_kmh)

    # 6. PEDESTRIAN WARNING
    image_np, flagPerson, areaPerson, areaDetails = estimate_stepping(
        person_boxes, image_np, flagPerson, areaPerson, areaDetails)

    # 7. HUD
    cv2.putText(image_np, "MULTIMODAL ADAS",
                (10, 25), font, 0.8, (0, 255, 0), 2)
    cv2.putText(image_np,
                f"Vehicles:{len(car_boxes)} | People:{len(person_boxes)}",
                (10, 55), font, 0.65, (200, 200, 200), 1)

    return image_np

# ─────────────────────────────────────────────────────────────────
#  DAY/NIGHT DETECTION
# ─────────────────────────────────────────────────────────────────
def confirm_day_or_night(frame, ctr):
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(hsv, (0,0,0), (180,255,35))
    mask    = cv2.dilate(cv2.erode(mask, None, 2), None, 2)
    ratio   = np.sum(mask == 0) / mask.size
    return ctr + (1 if ratio < 0.5 else -1)

# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    default="0")
    parser.add_argument("--speed",    type=float, default=40.0)
    parser.add_argument("--no-lanes", action="store_true")
    args = parser.parse_args()

    src = int(args.video) if args.video == "0" else args.video
    cap = cv2.VideoCapture(src)

    flag_night_counter = 0
    for _ in range(10):
        grabbed, frame = cap.read()
        if not grabbed: break
        flag_night_counter = confirm_day_or_night(
            imutils.resize(frame, width=1280), flag_night_counter)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    is_night = flag_night_counter > 0
    print(f"[INFO] Mode: {'NIGHT' if is_night else 'DAY'}")
    print(f"[INFO] Lanes: {'DISABLED (night)' if is_night else 'ENABLED'}")

    cv2.namedWindow("MULTIMODAL ADAS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MULTIMODAL ADAS", 1280, 720)

    fps = FPS().start()
    while True:
        grabbed, frame = cap.read()
        if not grabbed: break
        frame  = imutils.resize(frame, width=1280)
        result = run_multimodal(frame, args.speed,
                                run_lanes=(not args.no_lanes and not is_night))
        fps.update()
        try:
            curr_fps = fps.fps() if fps._numFrames > 1 else 0.0
        except Exception:
            curr_fps = 0.0
        cv2.putText(result, f"FPS:{curr_fps:.1f}", (1150, 25), font, 0.7, (0,255,0), 2)
        cv2.imshow("MULTIMODAL ADAS", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    try:
        print(f"[INFO] Elapsed:{fps.elapsed():.2f}s | FPS:{fps.fps():.2f}")
    except Exception:
        print("[INFO] Done.")
    cap.release()
    cv2.destroyAllWindows()
