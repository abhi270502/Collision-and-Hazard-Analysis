
# Collision and Hazard Analysis

Integrated driver-assistance style analysis pipeline for dashcam video.

This cleaned build keeps only the requested four features:

1. Object detection (YOLOv8)
2. Collision warning
3. Lane detection and lane-assist warning
4. Traffic signal tracking (red/clear-to-drive messaging)

## Run

Use the project virtual environment and run the integrated entry point:

```bash
./vehicle_warn_env/bin/python integrateyolo.py
```

Optional arguments:

```bash
./vehicle_warn_env/bin/python integrateyolo.py \
	--video IMG_3985.mp4 \
	--model yolov8n.pt \
	--width 1280 \
	--start-frame 0
```

## Runtime flow

1. The script asks you to select your dash region (to suppress own-vehicle detections).
2. The script asks you to select lane search ROI.
3. The integrated pipeline starts and shows warnings on the video stream.

Press `q` to quit.

## Feature Mapping

- Object detection: `integrateyolo.py` (YOLOv8 detections)
- Collision warning: `utils/estimate_collide_utils.py`
- Lane assist: `utils/lane_detection_utils.py`
- Traffic signal tracking: `utils/signalDetection_utils.py`
- Vehicle tracking overlay: `utils/tracking_utils.py`

## Notes

- The current build is focused on daytime integrated processing for the four requested capabilities.
- Legacy standalone scripts still present in subfolders can be removed later if you want a stricter minimal repository layout.