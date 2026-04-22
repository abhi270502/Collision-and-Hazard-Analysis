#!/usr/bin/env python3
"""
BDD100K JSON → YOLO TXT Converter (with Video Frame Extraction)

For your setup:
- Uses COCO class IDs directly (car=2, motorcycle=3, bus=5, truck=7, traffic light=9)
- Processes ALL frames from each video JSON (not just frame[0])
- Extracts actual video frames from .mov to .jpg
- Creates proper train/val split
- Generates data.yaml

Usage:
    python bdd100k_to_yolo_full.py \
        --bdd-root /path/to/bdd100k \
        --videos-subdir videos/train \
        --labels-subdir labels/det_20 \
        --output-dir yolo_dataset \
        --val-ratio 0.2 \
        --max-clips 100
"""

import argparse
import json
import random
import os
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python not installed. Install with: pip install opencv-python")
    exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Install with: pip install pyyaml")
    exit(1)


# ── COCO Class IDs (for your yolov8n.pt setup) ────────────────────────────────
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
}

# ── BDD100K category → COCO class ID ───────────────────────────────────────────
# Only these classes will be converted. Others are skipped.
BDD100K_TO_COCO = {
    "person":        0,
    "bicycle":       1,
    "car":           2,
    "motorcycle":    3,    # BDD100K calls it "motor"
    "bus":           5,
    "train":         6,
    "truck":         7,
    "traffic light": 9,
    # Skip: "rider" (no COCO equivalent)
    # Skip: "traffic sign" (no COCO equivalent, only stop sign at 11)
}

# Categories to completely skip (poly2d only, no box2d)
SKIP_CATEGORIES = {
    "area/drivable",
    "area/alternative",
    "lane/road curb",
    "lane/single white",
    "lane/double white",
    "lane/single yellow",
    "lane/double yellow",
    "lane/crosswalk",
    "lane/other",
}

IMG_W = 1280
IMG_H = 720


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BDD100K to YOLO format with video extraction.")
    parser.add_argument("--bdd-root", type=Path, default=Path("bdd100k"), help="BDD100K root folder.")
    parser.add_argument("--videos-subdir", default="videos/train", help="Relative video folder.")
    parser.add_argument("--labels-subdir", default="labels/", help="Relative JSON label folder.")
    parser.add_argument("--output-dir", type=Path, default=Path("yolo_dataset"), help="Output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg", help="Image extension.")
    parser.add_argument("--max-clips", type=int, default=None, help="Max clips to convert (for testing).")
    parser.add_argument("--max-frames-per-clip", type=int, default=None, help="Max frames per clip.")
    parser.add_argument("--skip-empty-labels", action="store_true", help="Skip frames with no objects.")
    return parser.parse_args()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box2d_to_yolo(x1, y1, x2, y2, width, height):
    """Convert absolute x1y1x2y2 to normalized cx cy w h."""
    x1 = clamp(float(x1), 0.0, float(width))
    y1 = clamp(float(y1), 0.0, float(height))
    x2 = clamp(float(x2), 0.0, float(width))
    y2 = clamp(float(y2), 0.0, float(height))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height

    return (cx, cy, w, h)


def split_stems(label_files, val_ratio, seed):
    """Split video IDs into train/val."""
    stems = [p.stem for p in label_files]
    rng = random.Random(seed)
    rng.shuffle(stems)
    val_count = int(len(stems) * val_ratio)
    return set(stems[val_count:]), set(stems[:val_count])


def ensure_dirs(output_dir):
    """Create directory structure."""
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_clip(json_path, split, videos_dir, output_dir, image_ext, 
                 max_frames_per_clip, skip_empty_labels):
    """Convert one video's JSON to YOLO frames."""
    stem = json_path.stem

    # Try .mov, then .mp4
    video_path = videos_dir / f"{stem}.mov"
    if not video_path.exists():
        video_path = videos_dir / f"{stem}.mp4"
    if not video_path.exists():
        return {"status": "missing_video", "stem": stem, "frames": 0}

    # Load JSON labels
    try:
        with open(json_path, "r") as f:
            label_data = json.load(f)
    except Exception as e:
        return {"status": "json_load_failed", "stem": stem, "frames": 0}

    frames = label_data.get("frames", [])
    if not frames:
        return {"status": "empty_labels", "stem": stem, "frames": 0}

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "video_open_failed", "stem": stem, "frames": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        return {"status": "bad_video_shape", "stem": stem, "frames": 0}

    written = 0
    img_dir = output_dir / "images" / split
    lbl_dir = output_dir / "labels" / split

    # Process each frame with a timestamp
    for frame_idx, frame_ann in enumerate(frames):
        if max_frames_per_clip and written >= max_frames_per_clip:
            break

        ts_ms = frame_ann.get("timestamp")
        if ts_ms is None:
            continue

        # Seek to frame using timestamp
        target_frame_idx = int(round((float(ts_ms) / 1000.0) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ok, image = cap.read()

        if not ok or image is None:
            continue

        # Process objects
        objects = frame_ann.get("objects", [])
        yolo_lines = []

        for obj in objects:
            cat = obj.get("category")

            # Skip non-detection objects
            if cat in SKIP_CATEGORIES or cat not in BDD100K_TO_COCO:
                continue

            box2d = obj.get("box2d")
            if not box2d:
                continue

            coco_id = BDD100K_TO_COCO[cat]
            yolo_coords = box2d_to_yolo(box2d["x1"], box2d["y1"], 
                                        box2d["x2"], box2d["y2"], 
                                        width, height)

            if yolo_coords is None:
                continue

            cx, cy, w, h = yolo_coords
            yolo_lines.append(f"{coco_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if skip_empty_labels and not yolo_lines:
            continue

        # Save image and label
        out_stem = f"{stem}_ts{int(ts_ms):06d}"
        img_path = img_dir / f"{out_stem}.{image_ext}"
        lbl_path = lbl_dir / f"{out_stem}.txt"

        cv2.imwrite(str(img_path), image)
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        written += 1

    cap.release()
    return {"status": "ok", "stem": stem, "frames": written}


def write_data_yaml(root_dir, output_dir):
    """Generate data.yaml for YOLO training."""
    data = {
        "path": str(root_dir.resolve()),
        "train": str((output_dir / "images" / "train").resolve()),
        "val": str((output_dir / "images" / "val").resolve()),
        "nc": len(COCO_CLASSES),
        "names": {i: COCO_CLASSES[i] for i in sorted(COCO_CLASSES.keys())},
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def main():
    args = parse_args()

    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("val-ratio must be in [0, 1)")

    root = args.bdd_root
    videos_dir = root / args.videos_subdir
    labels_dir = root / args.labels_subdir
    output_dir = args.output_dir

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    label_files = sorted([p for p in labels_dir.glob("*.json")])
    if not label_files:
        raise RuntimeError(f"No JSON files in {labels_dir}")

    if args.max_clips:
        label_files = label_files[:args.max_clips]

    print(f"Found {len(label_files)} JSON label files")
    print(f"Classes: {list(COCO_CLASSES.values())}")

    ensure_dirs(output_dir)
    train_stems, val_stems = split_stems(label_files, args.val_ratio, args.seed)

    summary = {
        "converted": 0,
        "missing_video": 0,
        "failed": 0,
        "train_frames": 0,
        "val_frames": 0,
    }

    for json_path in tqdm(label_files, desc="Converting"):
        stem = json_path.stem
        split = "val" if stem in val_stems else "train"

        result = convert_clip(
            json_path, split, videos_dir, output_dir,
            args.image_ext, args.max_frames_per_clip, args.skip_empty_labels
        )

        if result["status"] == "ok":
            summary["converted"] += 1
            summary[f"{split}_frames"] += result["frames"]
        elif result["status"] == "missing_video":
            summary["missing_video"] += 1
        else:
            summary["failed"] += 1

    yaml_path = write_data_yaml(root, output_dir)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output: {output_dir}")
    print(f"data.yaml: {yaml_path}")
    print(f"{'='*60}")
    print(f"Converted videos: {summary['converted']}")
    print(f"Missing videos: {summary['missing_video']}")
    print(f"Failed: {summary['failed']}")
    print(f"Train frames: {summary['train_frames']}")
    print(f"Val frames: {summary['val_frames']}")


if __name__ == "__main__":
    main()
