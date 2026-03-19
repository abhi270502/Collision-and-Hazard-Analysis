import argparse
import copy
from pathlib import Path

import cv2
import imutils
import numpy as np
from imutils.video import FPS
from ultralytics import YOLO

from utils import estimate_collide_utils
from utils import lane_detection_utils
from utils import signalDetection_utils
from utils import tracking_utils


font = cv2.FONT_HERSHEY_SIMPLEX

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
TRAFFIC_LIGHT_CLASS_ID = 9
DETECTION_CLASS_IDS = sorted(VEHICLE_CLASS_IDS | {TRAFFIC_LIGHT_CLASS_ID})

# Global state for temporal warning smoothing across frames.
crash_count_frames = 0
signalCounter = -99999
flagSignal = [0] * 20
number = 0
prev_frame = []
flagLanes = [0] * 20
refPt = []


def click_and_crop(event, x, y, flags, param):
    global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])


def _flatten_nms_indexes(indexes):
    if indexes is None:
        return []
    arr = np.array(indexes).reshape(-1)
    return [int(i) for i in arr.tolist()]


def _get_default_video_path():
    root = Path(__file__).resolve().parent
    return str(root / "IMG_3985.mp4")


def yolo_infer(
    model,
    dashPointer,
    lanePointer,
    frame,
    imgsz,
    conf,
    iou,
    device,
    show_debug_windows=False,
):
    global crash_count_frames
    global signalCounter, flagSignal
    global prev_frame, number
    global flagLanes

    image_np = np.array(frame)
    lane_image = copy.deepcopy(image_np)

    mask = 255 * np.ones_like(image_np)
    vertices = np.array(dashPointer, np.int32)
    cv2.fillPoly(mask, [vertices], [0, 0, 0])
    image_np = cv2.bitwise_and(image_np, mask)

    prediction = model.predict(
        image_np,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=DETECTION_CLASS_IDS,
        device=device,
        verbose=False,
    )[0]

    boxesCars, confidencesCars = [], []
    boxesLights, confidencesLights = [], []

    if prediction.boxes is not None:
        xyxy = prediction.boxes.xyxy.cpu().numpy()
        confs = prediction.boxes.conf.cpu().numpy()
        class_ids = prediction.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(xyxy, confs, class_ids):
            x1, y1, x2, y2 = box
            x = int(max(0, x1))
            y = int(max(0, y1))
            w = int(max(0, x2 - x1))
            h = int(max(0, y2 - y1))
            if w == 0 or h == 0:
                continue

            if class_id in VEHICLE_CLASS_IDS and (w * h) >= 800:
                boxesCars.append([x, y, w, h])
                confidencesCars.append(float(conf))
            elif class_id == TRAFFIC_LIGHT_CLASS_ID:
                boxesLights.append([x, y, w, h])
                confidencesLights.append(float(conf))

    raw_indexes_lights = cv2.dnn.NMSBoxes(boxesLights, confidencesLights, 0.25, 0.4)
    raw_indexes_cars = cv2.dnn.NMSBoxes(boxesCars, confidencesCars, 0.25, 0.4)

    indexesLights = _flatten_nms_indexes(raw_indexes_lights)
    indexesCars = _flatten_nms_indexes(raw_indexes_cars)

    image_np, signalCounter, flagSignal = signalDetection_utils.signalDetection(
        indexesLights,
        boxesLights,
        image_np,
        signalCounter,
        flagSignal,
        show_debug=show_debug_windows,
    )
    image_np, prev_frame, number = tracking_utils.tracking(
        indexesCars, boxesCars, image_np, prev_frame, number, show_debug=show_debug_windows
    )
    image_np, crash_count_frames = estimate_collide_utils.estimate_collide(
        indexesCars, boxesCars, image_np, crash_count_frames
    )
    image_np, flagLanes = lane_detection_utils.draw_lines(
        lanePointer,
        dashPointer,
        lane_image,
        image_np,
        flagLanes,
        show_debug=show_debug_windows,
    )

    cv2.putText(image_np, "Integrated 4-feature mode", (20, 40), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return image_np


def selectRegions(image, text, flag):
    global refPt
    clone = copy.deepcopy(image)
    while True:
        key = cv2.waitKey(1) & 0xFF

        color = [0, 255, 255] if flag == 1 else [0, 255, 0]
        cv2.putText(image, text, (240, 30), font, 1.2, color, 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'r' to reset selection.", (330, 70), font, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(image, "Press 'd' when selection is done.", (265, 105), font, 1.0, color, 2, cv2.LINE_AA)

        for pt in range(len(refPt) - 1):
            pt1, pt2 = refPt[pt], refPt[pt + 1]
            cv2.line(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), [0, 255, 255], 3)

        cv2.imshow("ROI", image)
        if key == ord("r"):
            image = copy.deepcopy(clone)
            refPt = []
        elif key == ord("d"):
            if flag == 1:
                return 0
            if flag == 2 and len(refPt) > 2:
                return 0
        elif key == ord("q"):
            return 1


def day(
    model,
    video_path,
    start_frame=0,
    width=1280,
    frame_skip=1,
    imgsz=640,
    conf=0.35,
    iou=0.45,
    device="cpu",
    show_debug_windows=False,
):
    global refPt

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, image = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame from video source.")
    image = imutils.resize(image, width=width)

    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", click_and_crop)

    quit_flow = selectRegions(copy.deepcopy(image), "Click points to select your vehicle dash.", 1)
    dashPointer = refPt
    if len(dashPointer) <= 2:
        dashPointer = [[0, 0], [0, 0], [0, 0]]
    refPt = []
    print("For dash:", dashPointer)
    if quit_flow == 1:
        cap.release()
        return

    quit_flow = selectRegions(copy.deepcopy(image), "Click points to select lane search region.", 2)
    lanePointer = refPt
    if len(lanePointer) <= 2:
        lanePointer = [[114, 690], [502, 384], [819, 391], [1201, 695]]
    print("For lanes:", lanePointer)
    if quit_flow == 1:
        cap.release()
        return

    cv2.destroyWindow("ROI")

    fps = FPS().start()
    frame_idx = 0
    last_output = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        frame = imutils.resize(frame, width=width)
        if frame_skip > 1 and (frame_idx % frame_skip != 0):
            if last_output is None:
                display_frame = frame
            else:
                display_frame = last_output
            cv2.putText(
                display_frame,
                f"Fast mode: skipping {frame_skip - 1}/{frame_skip} frames",
                (20, 75),
                font,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Final result", display_frame)
        else:
            last_output = yolo_infer(
                model,
                dashPointer,
                lanePointer,
                frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                show_debug_windows=show_debug_windows,
            )
            cv2.imshow("Final result", last_output)

        key = cv2.waitKey(1) & 0xFF
        fps.update()
        if key == ord("q"):
            break

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collision and Hazard Analysis integrated runtime (4-feature mode)."
    )
    parser.add_argument(
        "--video",
        default=_get_default_video_path(),
        help="Path to input video file. Defaults to IMG_3985.mp4 in project root.",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "yolov8n.pt"),
        help="Path to YOLO model weights.",
    )
    parser.add_argument("--width", type=int, default=960, help="Frame resize width.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Run detection every N frames (1 = every frame, 2 = half-rate).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for YOLO (e.g., cpu, mps, 0).",
    )
    parser.add_argument(
        "--show-debug-windows",
        action="store_true",
        help="Show intermediate debug windows for lanes/tracking/signals.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")

    model = YOLO(args.model)
    day(
        model,
        args.video,
        start_frame=args.start_frame,
        width=args.width,
        frame_skip=args.frame_skip,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show_debug_windows=args.show_debug_windows,
    )
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
