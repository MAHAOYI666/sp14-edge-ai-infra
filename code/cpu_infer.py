#!/usr/bin/env python3
# dataset_stream_infer_eval_onnx_cpu_unified_labelmap.py
# Unified dataset stream inference + GT evaluation for:
#   - YOLOv8 ONNX (raw head decode + NMS)
#   - PicoDet ONNX (already postprocessed Nx6 dets)
#
# IMPORTANT: For PicoDet, model class order (training label_list) may differ from dataset order.
# This script adds a model->dataset label mapping ONLY for PicoDet.

import time
import threading
import queue
from pathlib import Path
from collections import deque, defaultdict

import cv2
import numpy as np
import psutil

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError(
        "onnxruntime not found. Install with:\n"
        "  pip3 install onnxruntime\n"
        f"Original error: {e}"
    )

# Optional YAML for PicoDet-style NormalizeImage config
try:
    import yaml
except Exception:
    yaml = None


# =========================
# Dataset labels (YOUR GT order)
# =========================
DATASET_NAMES = [
    "car", "bus", "person", "bike", "truck",
    "motor", "train", "rider", "traffic sign", "traffic light"
]

# PicoDet training label_list (YOUR training order)
PICODET_TRAIN_LABELS = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]


# ----------------------------
# GT + Metrics
# ----------------------------
def load_gt_yolo(label_path: Path, img_w: int, img_h: int):
    gts = []
    if not label_path.exists():
        return gts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except Exception:
            continue

        x1 = int(round((xc - bw / 2.0) * img_w))
        y1 = int(round((yc - bh / 2.0) * img_h))
        x2 = int(round((xc + bw / 2.0) * img_w))
        y2 = int(round((yc + bh / 2.0) * img_h))

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        gts.append((x1, y1, x2, y2, cls))
    return gts


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter) / float(union + 1e-9)


def match_per_class(preds, gts, iou_thres=0.5):
    preds_by_c = defaultdict(list)
    gts_by_c = defaultdict(list)
    for p in preds:
        preds_by_c[p[4]].append(p)
    for g in gts:
        gts_by_c[g[4]].append(g)

    out = {}
    for cls in set(list(preds_by_c.keys()) + list(gts_by_c.keys())):
        cls_preds = sorted(preds_by_c.get(cls, []), key=lambda x: x[5], reverse=True)
        cls_gts = gts_by_c.get(cls, [])
        gt_used = [False] * len(cls_gts)

        tp = 0
        fp = 0
        records = []

        for p in cls_preds:
            pb = (p[0], p[1], p[2], p[3])
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(cls_gts):
                if gt_used[j]:
                    continue
                gb = (g[0], g[1], g[2], g[3])
                v = iou_xyxy(pb, gb)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thres:
                gt_used[best_j] = True
                tp += 1
                records.append((p[5], 1))
            else:
                fp += 1
                records.append((p[5], 0))

        fn = int(len(cls_gts) - sum(gt_used))
        out[cls] = {"tp": tp, "fp": fp, "fn": fn, "records": records, "gt_count": len(cls_gts)}
    return out


def ap_from_records(records, gt_count):
    if gt_count == 0:
        return float("nan")
    if not records:
        return 0.0

    records = sorted(records, key=lambda x: x[0], reverse=True)
    tp = np.array([r[1] for r in records], dtype=np.float32)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / (gt_count + 1e-9)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


# ----------------------------
# YAML helpers (PicoDet mean/std/labels)
# ----------------------------
def _find_normalize_cfg(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "NormalizeImage" and isinstance(v, dict):
                return v
            found = _find_normalize_cfg(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _find_normalize_cfg(it)
            if found is not None:
                return found
    return None


def _find_label_list(obj):
    if isinstance(obj, dict):
        if "label_list" in obj and isinstance(obj["label_list"], list):
            return obj["label_list"]
        for v in obj.values():
            found = _find_label_list(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _find_label_list(it)
            if found is not None:
                return found
    return None


def load_cfg_optional(cfg_path: str):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    is_scale = True
    labels = None

    if not cfg_path:
        return mean, std, is_scale, labels

    if yaml is None:
        print("[WARN] pyyaml not installed; ignore --cfg and use default mean/std.")
        return mean, std, is_scale, labels

    p = Path(cfg_path)
    if not p.exists():
        print(f"[WARN] cfg not found: {cfg_path}; use default mean/std.")
        return mean, std, is_scale, labels

    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        norm = _find_normalize_cfg(cfg) or {}
        mean = np.array(norm.get("mean", mean.tolist()), dtype=np.float32)
        std = np.array(norm.get("std", std.tolist()), dtype=np.float32)
        is_scale = bool(norm.get("is_scale", True))
        labels = _find_label_list(cfg)
        return mean, std, is_scale, labels
    except Exception as e:
        print(f"[WARN] failed to parse cfg: {e}; use default mean/std.")
        return mean, std, is_scale, labels


# ----------------------------
# Label mapping for PicoDet
# ----------------------------
def _norm_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


_PICODET_SYNONYMS = {
    "pedestrian": "person",
    "person": "person",
    "motorcycle": "motor",
    "motor": "motor",
    "bicycle": "bike",
    "bike": "bike",
    "trafficlight": "traffic light",
    "traffic sign": "traffic sign",
    "traffic light": "traffic light",
}


def build_picodet_model_to_dataset_map(model_labels, dataset_names):
    """
    Returns:
      - model_to_dataset: np.ndarray[int], shape [num_model_classes]
      - dataset_name_to_id: dict
    Any unmapped label -> -1 (ignored in eval/display)
    """
    dataset_name_to_id = {_norm_label(n): i for i, n in enumerate(dataset_names)}

    model_to_dataset = []
    for i, ml in enumerate(model_labels):
        key = _norm_label(ml)
        key = _PICODET_SYNONYMS.get(key, key)  # normalize synonyms
        did = dataset_name_to_id.get(key, -1)
        model_to_dataset.append(did)

    return np.array(model_to_dataset, dtype=np.int32), dataset_name_to_id


# ----------------------------
# Utility
# ----------------------------
def list_images_sorted(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs


class Prefetcher(threading.Thread):
    def __init__(self, image_paths, in_w, in_h, qmax=4):
        super().__init__(daemon=True)
        self.image_paths = image_paths
        self.in_w = in_w
        self.in_h = in_h
        self.q = queue.Queue(maxsize=qmax)
        self.stop_flag = False

    def stop(self):
        self.stop_flag = True

    def run(self):
        for img_path in self.image_paths:
            if self.stop_flag:
                break
            orig_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if orig_bgr is None:
                continue
            resized_bgr = cv2.resize(orig_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            oh, ow = orig_bgr.shape[:2]
            self.q.put((img_path, orig_bgr, resized_bgr, rgb, ow, oh))
        self.q.put(None)


# ----------------------------
# ONNX I/O picking
# ----------------------------
def _is_int_dim(d):
    if isinstance(d, (int, np.integer)):
        return True
    if isinstance(d, str):
        return d.isdigit()
    return False


def pick_image_input(sess: ort.InferenceSession):
    candidates = []
    for inp in sess.get_inputs():
        shp = inp.shape
        if isinstance(shp, (list, tuple)) and len(shp) == 4:
            candidates.append(inp)

    if not candidates:
        return sess.get_inputs()[0]

    for name in ("image", "images", "input", "input_0", "x"):
        for c in candidates:
            if c.name == name:
                return c

    for c in candidates:
        shp = c.shape
        if len(shp) == 4 and shp[1] == 3:
            return c
        if len(shp) == 4 and shp[3] == 3:
            return c

    return candidates[0]


def infer_input_hw_and_layout(image_inp, fallback_hw: int):
    shp = image_inp.shape
    layout = "NCHW"

    if not (isinstance(shp, (list, tuple)) and len(shp) == 4):
        return fallback_hw, fallback_hw, layout

    if shp[1] == 3:
        layout = "NCHW"
        h = shp[2]
        w = shp[3]
    elif shp[3] == 3:
        layout = "NHWC"
        h = shp[1]
        w = shp[2]
    else:
        layout = "NCHW"
        h = shp[2]
        w = shp[3]

    def to_int_or_fallback(x, fb):
        if _is_int_dim(x):
            return int(x)
        return fb

    in_h = to_int_or_fallback(h, fallback_hw)
    in_w = to_int_or_fallback(w, fallback_hw)
    return in_w, in_h, layout


# ----------------------------
# NMS + YOLOv8 decode
# ----------------------------
def nms_xyxy(boxes, scores, iou_thres):
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        inter = iw * ih
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_yolov8_output(raw_out, in_w, in_h, conf_thres=0.25, nms_iou=0.5, top_k=200, num_classes=10):
    out = np.asarray(raw_out)

    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]

    if out.ndim == 2:
        if out.shape[0] in (4, 5, 4 + num_classes, 5 + num_classes) and out.shape[1] > out.shape[0]:
            out = out.T
    elif out.ndim == 3:
        out = out.reshape(-1, out.shape[-1])

    if out.ndim != 2 or out.shape[1] < 6:
        return []

    D = out.shape[1]
    box_xywh = out[:, 0:4].astype(np.float32)

    has_obj = (D == 5 + num_classes) or (D > 5 and (D - 5) == num_classes)
    has_no_obj = (D == 4 + num_classes) or (D > 4 and (D - 4) == num_classes)

    if has_obj:
        obj = out[:, 4].astype(np.float32)
        cls_scores = out[:, 5:5 + num_classes].astype(np.float32)
    elif has_no_obj:
        obj = None
        cls_scores = out[:, 4:4 + num_classes].astype(np.float32)
    else:
        if D >= 5 + num_classes:
            obj = out[:, D - (num_classes + 1)].astype(np.float32)
            cls_scores = out[:, D - num_classes:].astype(np.float32)
        elif D >= 4 + num_classes:
            obj = None
            cls_scores = out[:, D - num_classes:].astype(np.float32)
        else:
            return []

    def maybe_sigmoid(v):
        vmax = float(np.max(v)) if v.size else 0.0
        vmin = float(np.min(v)) if v.size else 0.0
        if vmax > 1.5 or vmin < -0.5:
            return sigmoid(v)
        return np.clip(v, 0.0, 1.0)

    cls_scores = maybe_sigmoid(cls_scores)
    if obj is not None:
        obj = maybe_sigmoid(obj)

    max_box = float(np.max(box_xywh)) if box_xywh.size else 0.0
    if max_box <= 1.5:
        box_xywh[:, 0] *= in_w
        box_xywh[:, 2] *= in_w
        box_xywh[:, 1] *= in_h
        box_xywh[:, 3] *= in_h

    x = box_xywh[:, 0]
    y = box_xywh[:, 1]
    w = box_xywh[:, 2]
    h = box_xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, in_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, in_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, in_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, in_h - 1)

    dets = []
    for c in range(num_classes):
        score_c = cls_scores[:, c]
        if obj is not None:
            score_c = score_c * obj

        idx = np.where(score_c >= conf_thres)[0]
        if idx.size == 0:
            continue

        b = boxes[idx]
        s = score_c[idx].astype(np.float32)

        keep = nms_xyxy(b, s, iou_thres=nms_iou)
        for k in keep:
            bx = b[k]
            sc = float(s[k])
            x1i, y1i, x2i, y2i = map(int, bx.tolist())
            if x2i <= x1i or y2i <= y1i:
                continue
            dets.append((x1i, y1i, x2i, y2i, c, sc))

    dets.sort(key=lambda x: x[5], reverse=True)
    if len(dets) > top_k:
        dets = dets[:top_k]
    return dets


# ----------------------------
# PicoDet output parsing + box mapping
# ----------------------------
def pick_det_nx6(outs):
    for o in outs:
        a = np.asarray(o)
        if a.ndim == 2 and a.shape[1] == 6:
            return a
        if a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 6:
            return a[0]
    return None


def convert_boxes_to_input_xyxy(boxes_xyxy, in_w, in_h, orig_w=None, orig_h=None):
    b = boxes_xyxy.astype(np.float32).copy()
    if b.size == 0:
        return b

    m = float(np.max(b))
    if m <= 1.5:
        b[:, [0, 2]] *= in_w
        b[:, [1, 3]] *= in_h
    elif m <= max(in_w, in_h) + 5:
        pass
    else:
        if orig_w is not None and orig_h is not None and orig_w > 0 and orig_h > 0:
            sx = in_w / float(orig_w)
            sy = in_h / float(orig_h)
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy

    b[:, 0::2] = np.clip(b[:, 0::2], 0, in_w - 1)
    b[:, 1::2] = np.clip(b[:, 1::2], 0, in_h - 1)
    return b


# ----------------------------
# Preprocess
# ----------------------------
def preprocess_yolo(rgb_uint8, input_scale: float, layout: str):
    x = rgb_uint8.astype(np.float32) * float(input_scale)
    if layout == "NCHW":
        x = np.transpose(x, (2, 0, 1))  # CHW
        x = np.expand_dims(x, axis=0)   # NCHW
    else:
        x = np.expand_dims(x, axis=0)   # NHWC
    return x.astype(np.float32)


def preprocess_picodet(resized_bgr, mean, std, is_scale, layout: str):
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if is_scale:
        rgb = rgb / 255.0
    rgb = (rgb - mean) / std  # HWC

    if layout == "NCHW":
        x = np.transpose(rgb, (2, 0, 1))  # CHW
        x = np.expand_dims(x, axis=0)     # NCHW
    else:
        x = np.expand_dims(rgb, axis=0)   # NHWC
    return x.astype(np.float32)


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to model.onnx (YOLOv8 or PicoDet)")
    ap.add_argument("--images", required=True, help="Images directory (sequence)")
    ap.add_argument("--labels", default=None, help="Labels directory (same stem as image), optional")
    ap.add_argument("--num-images", type=int, default=0, help="Limit number of images to process (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (2 means skip every other image)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for TP matching (evaluation)")
    ap.add_argument("--nms-iou", type=float, default=0.50, help="IoU threshold for NMS (YOLOv8 decode)")
    ap.add_argument("--topk", type=int, default=200, help="Top-K detections")
    ap.add_argument("--headless", type=int, default=0, help="1 disables imshow")
    ap.add_argument("--show-gt", type=int, default=1, help="1 draws GT boxes if labels provided")
    ap.add_argument("--show-pred", type=int, default=1, help="1 draws prediction boxes")
    ap.add_argument("--input-size", type=int, default=800, help="Fallback model input size if ONNX has dynamic dims")
    ap.add_argument("--input-scale", type=float, default=(1.0 / 255.0),
                    help="YOLOv8: scale factor applied to uint8 RGB. Default=1/255.")
    ap.add_argument("--num-classes", type=int, default=len(DATASET_NAMES), help="Number of classes (default 10)")

    ap.add_argument("--cfg", default=None, help="(Optional) PaddleDetection infer_cfg.yml for PicoDet mean/std/labels")
    ap.add_argument("--force", choices=["auto", "yolo", "picodet"], default="auto",
                    help="Force model type; default auto by inspecting outputs")

    args = ap.parse_args()

    onnx_path = str(Path(args.onnx).expanduser().resolve())
    images_dir = Path(args.images).expanduser().resolve()
    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else None
    num_classes = int(args.num_classes)

    print("Using ONNX:", onnx_path)
    print("Images dir:", images_dir)
    print("Labels dir:", labels_dir if labels_dir else "(none)")
    print("Dataset names:", DATASET_NAMES)
    print(f"num_classes={num_classes} conf={args.conf} eval_iou={args.iou} topk={args.topk}")

    all_images = list_images_sorted(images_dir)
    if not all_images:
        print("[ERROR] No images found.")
        return

    stride = max(1, int(args.stride))
    images = all_images[::stride]
    if args.num_images and args.num_images > 0:
        images = images[: args.num_images]

    print(f"Total images on disk: {len(all_images)} | after stride={stride}: {len(all_images[::stride])} | will process: {len(images)}")

    so = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    image_inp = pick_image_input(sess)
    in_w, in_h, layout = infer_input_hw_and_layout(image_inp, fallback_hw=int(args.input_size))

    print("\n[MODEL] inputs:")
    for i, inp in enumerate(sess.get_inputs()):
        print(f"  in[{i}] name={inp.name} shape={inp.shape} type={inp.type}")
    print(f"[MODEL] picked image input: {image_inp.name} | (W,H)=({in_w},{in_h}) | layout={layout}")

    out_metas = sess.get_outputs()
    print("[MODEL] outputs:")
    for om in out_metas:
        print(f"  - {om.name} shape={om.shape} type={om.type}")

    mean, std, is_scale, cfg_labels = load_cfg_optional(args.cfg)
    if args.cfg:
        print(f"[CFG] cfg={args.cfg}")
        print(f"[CFG] mean={mean.tolist()} std={std.tolist()} is_scale={is_scale}")
        if cfg_labels:
            print(f"[CFG] label_list size={len(cfg_labels)} example={cfg_labels[:min(5, len(cfg_labels))]}")

    # Build PicoDet label mapping:
    # prefer cfg label_list if available; otherwise use your provided PICODET_TRAIN_LABELS.
    model_labels_for_map = cfg_labels if (cfg_labels and isinstance(cfg_labels, list)) else PICODET_TRAIN_LABELS
    picodet_model_to_dataset, _ = build_picodet_model_to_dataset_map(model_labels_for_map, DATASET_NAMES)
    print("\n[PICODET] model label_list used for mapping:")
    for i, n in enumerate(model_labels_for_map):
        did = int(picodet_model_to_dataset[i]) if i < len(picodet_model_to_dataset) else -1
        dst = DATASET_NAMES[did] if 0 <= did < len(DATASET_NAMES) else "UNMAPPED"
        print(f"  model_id={i:2d} '{n}'  -> dataset_id={did:2d} '{dst}'")

    prefetch = Prefetcher(images, in_w=in_w, in_h=in_h, qmax=4)
    prefetch.start()

    proc = psutil.Process()
    psutil.cpu_percent(interval=None)

    fps_window = deque(maxlen=60)
    fps_smooth = 0.0
    cpu_percent = 0.0
    mem_mb = 0.0
    last_res_time = 0.0
    res_interval = 1.0

    # summary stats
    fps_sum = 0.0
    fps_min = float("inf")
    fps_max = 0.0
    fps_count = 0

    infer_sum = 0.0
    infer_min = float("inf")
    infer_max = 0.0
    infer_count = 0

    cpu_sum = 0.0
    cpu_min = float("inf")
    cpu_max = 0.0
    cpu_count = 0

    ram_sum = 0.0
    ram_min = float("inf")
    ram_max = 0.0
    ram_count = 0

    # Eval accumulators (dataset class space)
    cls_tp = np.zeros(num_classes, dtype=np.int64)
    cls_fp = np.zeros(num_classes, dtype=np.int64)
    cls_fn = np.zeros(num_classes, dtype=np.int64)
    cls_gt_count = np.zeros(num_classes, dtype=np.int64)
    cls_records = {c: [] for c in range(num_classes)}

    frame_id = 0
    first_frame = True

    if not args.headless:
        cv2.namedWindow("ONNX CPU Dataset Stream (Unified+LabelMap)", cv2.WINDOW_NORMAL)

    def dataset_name(cid: int) -> str:
        if 0 <= cid < len(DATASET_NAMES):
            return DATASET_NAMES[cid]
        return str(cid)

    model_type = args.force  # "auto"/"yolo"/"picodet"

    try:
        while True:
            loop_start = time.time()

            item = prefetch.q.get()
            if item is None:
                print("end (no more images)")
                break

            img_path, orig_bgr, resized_bgr, rgb_uint8, orig_w, orig_h = item
            frame_id += 1

            # Determine mode (auto via Nx6 output existence later)
            feed = {}

            # preprocess (auto heuristic using presence of paddle-style inputs)
            in_names = {i.name for i in sess.get_inputs()}
            likely_picodet = (("scale_factor" in in_names) or ("im_shape" in in_names) or (image_inp.name == "image"))

            if model_type == "yolo":
                x = preprocess_yolo(rgb_uint8, args.input_scale, layout)
            elif model_type == "picodet":
                x = preprocess_picodet(resized_bgr, mean, std, is_scale, layout)
            else:
                x = preprocess_picodet(resized_bgr, mean, std, is_scale, layout) if likely_picodet else preprocess_yolo(rgb_uint8, args.input_scale, layout)

            feed[image_inp.name] = x

            # extra inputs for PaddleDet family
            scale_y = float(in_h) / float(orig_h)
            scale_x = float(in_w) / float(orig_w)
            for extra in sess.get_inputs():
                if extra.name == image_inp.name:
                    continue
                if extra.name == "scale_factor":
                    feed[extra.name] = np.array([[scale_y, scale_x]], dtype=np.float32)
                elif extra.name == "im_shape":
                    feed[extra.name] = np.array([[float(in_h), float(in_w)]], dtype=np.float32)

            infer_start = time.time()
            outs = sess.run(None, feed)
            infer_end = time.time()

            infer_ms = (infer_end - infer_start) * 1000.0
            infer_sum += infer_ms
            infer_count += 1
            infer_min = min(infer_min, infer_ms)
            infer_max = max(infer_max, infer_ms)

            dt = time.time() - loop_start
            if dt > 0.0:
                fps_inst = 1.0 / dt
                fps_window.append(fps_inst)
                fps_smooth = sum(fps_window) / len(fps_window)
                fps_sum += fps_inst
                fps_count += 1
                fps_min = min(fps_min, fps_inst)
                fps_max = max(fps_max, fps_inst)

            if first_frame:
                first_frame = False
                print("\n[DEBUG] First frame outputs:")
                for i, o in enumerate(outs):
                    a = np.asarray(o)
                    print(f"  out[{i}] shape={a.shape} dtype={a.dtype}")

            # Parse outputs
            preds = []
            det_nx6 = pick_det_nx6(outs)
            if model_type == "picodet":
                use_picodet = True
            elif model_type == "yolo":
                use_picodet = False
            else:
                use_picodet = (det_nx6 is not None)

            if use_picodet:
                det = det_nx6
                if det is not None and det.size and det.shape[1] == 6:
                    cls_ids = det[:, 0].astype(np.int32)
                    scores = det[:, 1].astype(np.float32)
                    boxes = det[:, 2:6].astype(np.float32)

                    keep = scores >= float(args.conf)
                    cls_ids = cls_ids[keep]
                    scores = scores[keep]
                    boxes = boxes[keep]

                    boxes = convert_boxes_to_input_xyxy(boxes, in_w=in_w, in_h=in_h, orig_w=orig_w, orig_h=orig_h)

                    # *** KEY: model label -> dataset label mapping ***
                    for cid, sc, (x1, y1, x2, y2) in zip(cls_ids, scores, boxes):
                        cid = int(cid)
                        if cid < 0 or cid >= len(picodet_model_to_dataset):
                            continue
                        did = int(picodet_model_to_dataset[cid])
                        if did < 0 or did >= num_classes:
                            continue  # unmapped -> ignore

                        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                        if x2i <= x1i or y2i <= y1i:
                            continue
                        preds.append((x1i, y1i, x2i, y2i, did, float(sc)))
            else:
                raw_out = outs[0]
                preds = decode_yolov8_output(
                    raw_out=raw_out,
                    in_w=in_w,
                    in_h=in_h,
                    conf_thres=float(args.conf),
                    nms_iou=float(args.nms_iou),
                    top_k=int(args.topk),
                    num_classes=num_classes,
                )

            # GT + eval
            gts = []
            if labels_dir is not None:
                lbl_path = labels_dir / f"{img_path.stem}.txt"
                gts = load_gt_yolo(lbl_path, img_w=in_w, img_h=in_h)

                matched = match_per_class(preds, gts, iou_thres=float(args.iou))
                for c, m in matched.items():
                    if 0 <= c < num_classes:
                        cls_tp[c] += m["tp"]
                        cls_fp[c] += m["fp"]
                        cls_fn[c] += m["fn"]
                        cls_gt_count[c] += m["gt_count"]
                        cls_records[c].extend(m["records"])

            # resources
            now = time.time()
            if now - last_res_time >= res_interval:
                last_res_time = now
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_mb = proc.memory_info().rss / (1024.0 * 1024.0)

                cpu_sum += cpu_percent
                cpu_count += 1
                cpu_min = min(cpu_min, cpu_percent)
                cpu_max = max(cpu_max, cpu_percent)

                ram_sum += mem_mb
                ram_count += 1
                ram_min = min(ram_min, mem_mb)
                ram_max = max(ram_max, mem_mb)

            if args.headless:
                continue

            overlay = resized_bgr.copy()

            # Draw GT
            if args.show_gt and gts:
                for (x1, y1, x2, y2, cls) in gts:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        overlay,
                        f"GT:{cls}:{dataset_name(cls)}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            # Draw preds
            if args.show_pred and preds:
                for (x1, y1, x2, y2, cls_id, score) in preds:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        f"P:{cls_id}:{dataset_name(cls_id)} {score:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

            mode_str = "PicoDet(Nx6+LabelMap)" if use_picodet else "YOLOv8(decode)"
            lines = [
                f"Frame: {frame_id} | {img_path.name}",
                f"Mode: {mode_str}",
                f"FPS: {fps_smooth:5.1f}",
                f"Infer: {infer_ms:6.1f} ms",
                f"CPU: {cpu_percent:5.1f}%",
                f"RAM: {mem_mb:7.1f} MB",
                f"input={in_w}x{in_h} layout={layout} conf={args.conf:.2f} nms_iou={args.nms_iou:.2f} eval_iou={args.iou:.2f} topk={args.topk}",
            ]

            if labels_dir is not None:
                tp_all = int(cls_tp.sum())
                fp_all = int(cls_fp.sum())
                fn_all = int(cls_fn.sum())
                prec_all = tp_all / (tp_all + fp_all + 1e-9)
                rec_all = tp_all / (tp_all + fn_all + 1e-9)
                f1_all = (2 * prec_all * rec_all) / (prec_all + rec_all + 1e-9)
                lines.append(f"Eval so far P/R/F1: {prec_all:.3f}/{rec_all:.3f}/{f1_all:.3f}")

            panel_w = 1040
            panel_h = 26 + 18 * len(lines)
            cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)

            y0 = 22
            for i, text in enumerate(lines):
                y = y0 + i * 18
                cv2.putText(
                    overlay,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("ONNX CPU Dataset Stream (Unified+LabelMap)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        prefetch.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    # Summary
    print("\n========== Runtime summary ==========")
    print(f"Frames processed: {frame_id}")

    if fps_count > 0:
        fps_avg = fps_sum / fps_count
        print(f"FPS instant: avg={fps_avg:.2f}, min={fps_min:.2f}, max={fps_max:.2f}")
    else:
        print("FPS instant: no data")

    if infer_count > 0:
        infer_avg = infer_sum / infer_count
        print(f"Infer time (ms): avg={infer_avg:.2f}, min={infer_min:.2f}, max={infer_max:.2f}")
    else:
        print("Infer time (ms): no data")

    if cpu_count > 0:
        cpu_avg = cpu_sum / cpu_count
        print(f"CPU usage (%): avg={cpu_avg:.2f}, min={cpu_min:.2f}, max={cpu_max:.2f}")
    else:
        print("CPU usage (%): no data")

    if ram_count > 0:
        ram_avg = ram_sum / ram_count
        print(f"RAM usage (MB): avg={ram_avg:.2f}, min={ram_min:.2f}, max={ram_max:.2f}")
    else:
        print("RAM usage (MB): no data")

    if labels_dir is not None:
        print("\n========== Quantitative evaluation (IoU>=%.2f) ==========" % float(args.iou))

        ap_list = []
        valid_ap_classes = 0

        for c in range(num_classes):
            tp = int(cls_tp[c])
            fp = int(cls_fp[c])
            fn = int(cls_fn[c])
            gt = int(cls_gt_count[c])

            if gt == 0 and (tp + fp) == 0:
                continue

            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = (2 * prec * rec) / (prec + rec + 1e-9)

            ap50 = ap_from_records(cls_records[c], gt)
            ap_str = "nan" if np.isnan(ap50) else f"{ap50:.3f}"

            name = DATASET_NAMES[c] if 0 <= c < len(DATASET_NAMES) else f"class_{c}"
            print(f"[{c:2d}] {name:13s} | GT={gt:5d} TP={tp:5d} FP={fp:5d} FN={fn:5d} "
                  f"| P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f} | AP50={ap_str}")

            if not np.isnan(ap50):
                ap_list.append(ap50)
                valid_ap_classes += 1

        if valid_ap_classes > 0:
            map50 = float(np.mean(ap_list))
            print(f"\nmAP@0.5 over {valid_ap_classes} classes (with GT): {map50:.3f}")
        else:
            print("\nmAP@0.5: no valid classes (no GT found?)")

        tp_all = int(cls_tp.sum())
        fp_all = int(cls_fp.sum())
        fn_all = int(cls_fn.sum())
        prec_all = tp_all / (tp_all + fp_all + 1e-9)
        rec_all = tp_all / (tp_all + fn_all + 1e-9)
        f1_all = (2 * prec_all * rec_all) / (prec_all + rec_all + 1e-9)
        print(f"Overall P/R/F1 (micro): {prec_all:.3f}/{rec_all:.3f}/{f1_all:.3f}")

    print("=====================================\n")


if __name__ == "__main__":
    main()
