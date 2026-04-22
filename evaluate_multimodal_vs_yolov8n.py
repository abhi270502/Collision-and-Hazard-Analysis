import os, time, yaml, json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

OUT_DIR    = Path("output"); OUT_DIR.mkdir(exist_ok=True)
DATA_YAML  = "yolo_dataset/data.yaml"
VAL_IMAGES = Path("yolo_dataset/images/val")
VAL_LABELS = Path("yolo_dataset/labels/val")

OLD_MODEL = "yolov8n.pt"
VEH_MODEL = "models_and_weights/vehicle_elite.pt"
SGN_MODEL = "models_and_weights/traffic_signs.pt"
LGT_MODEL = "models_and_weights/best.pt"

IMG_SIZE   = 640
CONF_TH    = 0.001
CONF_LIGHT = 0.15
IOU_MATCH  = 0.5
MAX_IMAGES = None

# Fair eval: only classes that BOTH systems can predict
# 0=person, 2=car (vehicle), 9=traffic light
TARGET_IDS = {0, 2, 9}

with open(DATA_YAML) as f: data_cfg = yaml.safe_load(f)
nm = data_cfg['names']
gt_names = {int(k):v for k,v in nm.items()} if isinstance(nm, dict) \
           else {i:n for i,n in enumerate(nm)}

old_model = YOLO(OLD_MODEL)
veh_model = YOLO(VEH_MODEL)
sgn_model = YOLO(SGN_MODEL)
lgt_model = YOLO(LGT_MODEL)

def norm(s): return str(s).strip().lower().replace('_', ' ')

# ── Class mappers ──────────────────────────────────────────────────────────────
def map_old(name):
    n = norm(name)
    return {'person':0,'bicycle':1,'car':2,'motorcycle':3,'motorbike':3,
            'bus':5,'truck':7,'traffic light':9,'stop sign':11}.get(n)

def map_veh(name):
    n = norm(name)
    if n == 'pedestrians': return 0   # → person
    if n == 'vehicles':    return 2   # → car (generic vehicle)
    return None                       # signpost → skip

def map_lgt(name):
    n = norm(name)
    if n in {'red', 'green', 'yellow'}: return 9  # → traffic light
    return None

# ── Geometry ───────────────────────────────────────────────────────────────────
def yolo_xyxy(parts, w, h):
    _, xc, yc, bw, bh = [float(x) for x in parts]
    return [(xc - bw/2)*w, (yc - bh/2)*h, (xc + bw/2)*w, (yc + bh/2)*h]

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    aA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    aB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aA + aB - inter) if (aA + aB - inter) > 0 else 0.0

# ── GT loader ─────────────────────────────────────────────────────────────────
def load_gt(lp, w, h):
    gts = []
    if not lp.exists(): return gts
    for raw in lp.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) != 5: continue
        cid = int(float(parts[0]))
        if cid not in TARGET_IDS: continue
        gts.append({'class_id': cid, 'bbox': yolo_xyxy(parts, w, h)})
    return gts

# ── Inference ─────────────────────────────────────────────────────────────────
def infer_old(img):
    t0  = time.perf_counter()
    res = old_model(img, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False)[0]
    dt  = time.perf_counter() - t0
    preds = []
    for b in res.boxes:
        cid = map_old(old_model.names[int(b.cls)])
        if cid is None or cid not in TARGET_IDS: continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        preds.append({'class_id': cid, 'bbox': [x1,y1,x2,y2], 'conf': float(b.conf)})
    return preds, dt

def infer_new(img):
    preds = []
    t0 = time.perf_counter()
    r1 = veh_model(img, imgsz=IMG_SIZE, conf=CONF_TH,    verbose=False)[0]
    t1 = time.perf_counter()
    r2 = sgn_model(img, imgsz=IMG_SIZE, conf=CONF_TH,    verbose=False)[0]
    t2 = time.perf_counter()
    r3 = lgt_model(img, imgsz=IMG_SIZE, conf=CONF_LIGHT, verbose=False)[0]
    t3 = time.perf_counter()

    for b in r1.boxes:
        cid = map_veh(veh_model.names[int(b.cls)])
        if cid is None or cid not in TARGET_IDS: continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        preds.append({'class_id': cid, 'bbox': [x1,y1,x2,y2], 'conf': float(b.conf)})

    for b in r3.boxes:
        cid = map_lgt(lgt_model.names[int(b.cls)])
        if cid is None or cid not in TARGET_IDS: continue
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        preds.append({'class_id': cid, 'bbox': [x1,y1,x2,y2], 'conf': float(b.conf)})

    return preds, (t3 - t0), (t1-t0, t2-t1, t3-t2)

# ── Core evaluator ────────────────────────────────────────────────────────────
def evaluate(image_paths, mode='old'):
    stats = {cid: {'tp':0,'fp':0,'fn':0,'num_gt':0} for cid in sorted(TARGET_IDS)}
    infer_times = []
    sub_t = defaultdict(list)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        gts  = load_gt(VAL_LABELS / f"{img_path.stem}.txt", w, h)
        for g in gts: stats[g['class_id']]['num_gt'] += 1

        if mode == 'old':
            preds, dt = infer_old(img)
            infer_times.append(dt)
        else:
            preds, dt, (vt, st, lt) = infer_new(img)
            infer_times.append(dt)
            sub_t['veh_ms'].append(vt * 1000)
            sub_t['sign_ms'].append(st * 1000)
            sub_t['light_ms'].append(lt * 1000)

        pby = defaultdict(list); gby = defaultdict(list)
        for i, p in enumerate(preds): pby[p['class_id']].append((i, p))
        for i, g in enumerate(gts):   gby[g['class_id']].append((i, g))

        for cid in sorted(TARGET_IDS):
            pl   = sorted(pby.get(cid, []), key=lambda x: x[1]['conf'], reverse=True)
            gl   = gby.get(cid, [])
            used = set()
            for _, p in pl:
                best_iou, best_gi = 0, None
                for gi, g in gl:
                    if gi in used: continue
                    ov = iou(p['bbox'], g['bbox'])
                    if ov > best_iou: best_iou = ov; best_gi = gi
                if best_iou >= IOU_MATCH and best_gi is not None:
                    stats[cid]['tp'] += 1; used.add(best_gi)
                else:
                    stats[cid]['fp'] += 1
            for gi, _ in gl:
                if gi not in used: stats[cid]['fn'] += 1

    rows = []
    for cid in sorted(TARGET_IDS):
        s = stats[cid]; tp, fp, fn = s['tp'], s['fp'], s['fn']
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({'class_id': cid, 'class_name': gt_names.get(cid, str(cid)),
                     'tp': tp, 'fp': fp, 'fn': fn,
                     'num_gt': s['num_gt'],          # ← renamed: avoids df.gt clash
                     'precision': prec, 'recall': rec, 'f1': f1, 'mode': mode})

    df = pd.DataFrame(rows)
    avg_ms = np.mean(infer_times) * 1000 if infer_times else 0
    fps    = 1000 / avg_ms if avg_ms > 0 else 0

    mtp = int(df['tp'].sum())
    mfp = int(df['fp'].sum())
    mfn = int(df['fn'].sum())
    mp  = mtp / (mtp + mfp) if (mtp + mfp) else 0
    mr  = mtp / (mtp + mfn) if (mtp + mfn) else 0
    mf  = 2*mp*mr / (mp + mr) if (mp + mr) else 0

    summary = {
        'mode': mode, 'images': len(image_paths),
        'micro_p': mp, 'micro_r': mr, 'micro_f1': mf,
        'macro_p': float(df['precision'].mean()),
        'macro_r': float(df['recall'].mean()),
        'macro_f1': float(df['f1'].mean()),
        'avg_ms': round(avg_ms, 3), 'fps': round(fps, 2),
        'total_gt': int(df['num_gt'].sum()),        # ← df['num_gt'] not df.gt
        'total_tp': mtp, 'total_fp': mfp, 'total_fn': mfn,
    }
    if sub_t:
        for k, v in sub_t.items():
            summary[k] = round(float(np.mean(v)), 2)
    return df, summary

# ── Run ───────────────────────────────────────────────────────────────────────
image_paths = sorted(
    list(VAL_IMAGES.glob('*.jpg')) +
    list(VAL_IMAGES.glob('*.png')) +
    list(VAL_IMAGES.glob('*.jpeg'))
)
if MAX_IMAGES:
    image_paths = image_paths[:MAX_IMAGES]

print(f"[INFO] Evaluating on {len(image_paths)} images")
print(f"[INFO] Fair eval classes: {[gt_names[c] for c in sorted(TARGET_IDS)]}")

old_df, old_sum = evaluate(image_paths, 'old')
new_df, new_sum = evaluate(image_paths, 'new')

comp = old_df.merge(new_df, on=['class_id', 'class_name'], suffixes=('_old', '_new'))
comp['prec_delta'] = comp['precision_new'] - comp['precision_old']
comp['rec_delta']  = comp['recall_new']    - comp['recall_old']
comp['f1_delta']   = comp['f1_new']        - comp['f1_old']

comp.to_csv(OUT_DIR / 'class_comparison.csv', index=False)
old_df.to_csv(OUT_DIR / 'old_model_class_metrics.csv', index=False)
new_df.to_csv(OUT_DIR / 'updated_model_class_metrics.csv', index=False)
pd.DataFrame([old_sum, new_sum]).to_csv(OUT_DIR / 'evaluation_summary.csv', index=False)
with open(OUT_DIR / 'evaluation_summary.json', 'w') as f:
    json.dump({'old': old_sum, 'new': new_sum}, f, indent=2)

print("\n=== SUMMARY (Fair: person | car | traffic light) ===")
print(f"{'Metric':<22} {'Old yolov8n':>14} {'New Multimodal':>15}")
print("-" * 53)
for k in ['micro_p','micro_r','micro_f1','macro_p','macro_r','macro_f1']:
    print(f"{k:<22} {old_sum[k]:>14.4f} {new_sum[k]:>15.4f}")
print(f"{'FPS':<22} {old_sum['fps']:>14.1f} {new_sum['fps']:>15.1f}")
print(f"{'Avg latency (ms)':<22} {old_sum['avg_ms']:>14.1f} {new_sum['avg_ms']:>15.1f}")
print("\n=== PER-CLASS ===")
print(comp[['class_name',
            'precision_old','precision_new','prec_delta',
            'recall_old','recall_new','rec_delta',
            'f1_old','f1_new','f1_delta']].to_string(index=False))
print("\nSaved all CSVs to output/")