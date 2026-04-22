# BDD100K → YOLO Conversion & Evaluation Pipeline

Complete pipeline for converting BDD100K dataset to YOLO format and evaluating object detection models.

## Setup

### Prerequisites
```bash
pip install opencv-python pyyaml ultralytics tqdm torch torchvision
```

### Directory Structure
```
bdd100k/
├── videos/
│   ├── train/
│   │   ├── 00a0f008-3c67908e.mov
│   │   ├── 00a1b8aa-c0ba5a4e.mov
│   │   └── ...
│   └── val/
└── labels/
    ├── det_20/
    │   ├── 00a0f008-3c67908e.json
    │   ├── 00a1b8aa-c0ba5a4e.json
    │   └── ...

yolo_dataset/
├── images/
│   ├── train/
│   │   ├── video1_ts010000.jpg
│   │   ├── video1_ts010050.jpg
│   │   └── ...
│   └── val/
├── labels/
│   ├── train/
│   │   ├── video1_ts010000.txt
│   │   ├── video1_ts010050.txt
│   │   └── ...
│   └── val/
├── data.yaml
└── metrics/
```

---

## Step 1: Convert BDD100K JSON to YOLO TXT

### Basic Conversion
```bash
python bdd100k_to_yolo_full.py \
    --bdd-root /path/to/bdd100k \
    --videos-subdir videos/train \
    --labels-subdir labels/det_20 \
    --output-dir yolo_dataset \
    --val-ratio 0.2
```

### With Options
```bash
python bdd100k_to_yolo_full.py \
    --bdd-root bdd100k \
    --videos-subdir videos/train \
    --labels-subdir labels/det_20 \
    --output-dir yolo_dataset \
    --val-ratio 0.2 \
    --seed 42 \
    --max-clips 100 \
    --max-frames-per-clip 30 \
    --skip-empty-labels
```

### Class Mapping (COCO IDs)
The script uses these COCO class IDs:
```
0: person
1: bicycle
2: car
3: motorcycle
5: bus
6: train
7: truck
9: traffic light
```

### Output
- `yolo_dataset/images/train/*.jpg` — Training images
- `yolo_dataset/images/val/*.jpg` — Validation images
- `yolo_dataset/labels/train/*.txt` — Training labels (YOLO format)
- `yolo_dataset/labels/val/*.txt` — Validation labels
- `yolo_dataset/data.yaml` — Dataset config for YOLO training

Each `.txt` label file contains one object per line:
```
class_id cx cy w h
2 0.548 0.519 0.139 0.164
0 0.115 0.411 0.043 0.262
9 0.488 0.186 0.020 0.077
```

Where:
- `class_id` = 0-9 (COCO class)
- `cx, cy` = center coordinates (normalized 0-1)
- `w, h` = width, height (normalized 0-1)

---

## Step 2: Train YOLO Model

### Using Pre-trained yolov8n.pt
```bash
yolo detect train \
    data=yolo_dataset/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

### Output
```
runs/detect/train/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── confusion_matrix.png
└── ...
```

---

## Step 3: Evaluate Model

### Quick Evaluation
```bash
python evaluate_yolo.py \
    --model runs/detect/train/weights/best.pt \
    --data yolo_dataset/data.yaml \
    --conf 0.25 \
    --iou 0.45 \
    --device 0
```

### Advanced Evaluation
```bash
python evaluate_advanced.py \
    --model runs/detect/train/weights/best.pt \
    --images yolo_dataset/images/val \
    --labels yolo_dataset/labels/val \
    --output eval_results \
    --conf 0.25 \
    --iou 0.5
```

### Output Metrics
```
Overall Metrics (IoU=0.5)
Precision: 0.8234
Recall:    0.7891
F1-Score:  0.8060
TP: 1245, FP: 267, FN: 324
```

---

## Class ID Reference

| COCO ID | Class Name | Included in BDD100K |
|---------|------------|-------------------|
| 0 | person | ✓ |
| 1 | bicycle | ✓ |
| 2 | car | ✓ |
| 3 | motorcycle | ✓ (BDD100K "motor") |
| 4 | airplane | ✗ |
| 5 | bus | ✓ |
| 6 | train | ✓ |
| 7 | truck | ✓ |
| 8 | boat | ✗ |
| 9 | traffic light | ✓ |
| 10-79 | other COCO classes | ✗ |

**Note:** BDD100K classes `rider` and `traffic sign` are NOT in COCO and cannot be detected with `yolov8n.pt`.

---

## Understanding the YOLO Format

### Input (BDD100K JSON)
```json
{
  "name": "00a0f008-3c67908e",
  "frames": [
    {
      "timestamp": 10000,
      "objects": [
        {
          "category": "car",
          "id": 17,
          "box2d": {
            "x1": 837.762,
            "y1": 237.682,
            "x2": 926.928,
            "y2": 273.936
          }
        }
      ]
    }
  ]
}
```

### Conversion Process
1. Read BDD100K `box2d` absolute coordinates: `x1, y1, x2, y2`
2. Convert to YOLO normalized format:
   - `cx = (x1 + x2) / (2 * img_width)`
   - `cy = (y1 + y2) / (2 * img_height)`
   - `w = (x2 - x1) / img_width`
   - `h = (y2 - y1) / img_height`
3. Map BDD100K category ("car") to YOLO COCO class ID (2)
4. Write to `.txt` file: `2 0.548 0.519 0.139 0.164`

### Output (YOLO Format)
```
2 0.548 0.519 0.139 0.164
0 0.115 0.411 0.043 0.262
9 0.488 0.186 0.020 0.077
```

---

## Troubleshooting

### Missing Classes
If you see warnings like "Unknown category 'rider'", this is expected. BDD100K has classes that don't exist in COCO:
- `rider` — no COCO equivalent (only "person" is detected)
- `traffic sign` — no COCO equivalent (only "stop sign" at ID 11)

These objects are silently skipped during conversion.

### Empty Label Files
Some frames may have no detectable objects. Empty `.txt` files are created for these frames. This is normal.

### Video Extraction Issues
If `.mov` files aren't found, the script tries `.mp4` as fallback. Make sure video files exist and are readable.

### Class ID Mismatch
The `data.yaml` generated by the converter uses COCO class IDs (0-79). Make sure your trained model matches this mapping.

---

## Performance Notes

- **Conversion speed:** ~100-200 videos per minute (depends on video length and I/O)
- **Typical dataset size:** 100K videos → 3-5M frames → 200-500GB images (at 1280×720 JPG)
- **Training time:** YOLOv8n on 100 epochs, 640×640, batch 16: ~6-12 hours on single GPU

---

## Next Steps

1. ✓ Convert BDD100K to YOLO format
2. ✓ Train YOLOv8 model
3. ✓ Evaluate on validation set
4. Integrate with Ashish's warning system
5. Add TTC calculation from ravesandstorm repo
