#!/usr/bin/env python3
# dataset_stream_infer_eval.py
# Sequential image stream inference + GT evaluation for Hailo HEF.
# Adds: --num-images to limit how many images to process.
# Adds: optional Python-side NMS controlled by CLI.

import time
import threading
import queue
from pathlib import Path
from collections import deque, defaultdict

import cv2
import numpy as np
import psutil
import hailo_platform as hpf


SP14_NAMES = [
    "car", "bus", "person", "bike", "truck",
    "motor", "train", "rider", "traffic sign", "traffic light"
]


def postprocess_hailo_nms(nms_output, img_w, img_h, conf_thres=0.25, top_k=200):
    """
    This function only DE-CODES the model output into a flat list:
      (x1, y1, x2, y2, cls_id, score)
    It does NOT perform NMS suppression.
    """
    detections = []

    if isinstance(nms_output, list):
        for cls_id, cls_dets in enumerate(nms_output):
            if not isinstance(cls_dets, np.ndarray) or cls_dets.size == 0:
                continue
            for det in cls_dets:
                if det.shape[0] < 5:
                    continue
                ymin, xmin, ymax, xmax, score = det[:5]
                score = float(score)
                if score < conf_thres:
                    continue
                x1 = int(xmin * img_w)
                y1 = int(ymin * img_h)
                x2 = int(xmax * img_w)
                y2 = int(ymax * img_h)
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append((x1, y1, x2, y2, cls_id, score))

    elif isinstance(nms_output, np.ndarray):
        # Common case in some Hailo postprocess outputs: [num_classes, 5, num_dets] or [num_classes, >=5, num_dets]
        if nms_output.ndim == 3 and nms_output.shape[1] >= 5:
            num_classes, _, num_dets = nms_output.shape
            for cls_id in range(num_classes):
                for det_id in range(num_dets):
                    ymin, xmin, ymax, xmax, score = nms_output[cls_id, 0:5, det_id]
                    score = float(score)
                    if score < conf_thres:
                        continue
                    x1 = int(xmin * img_w)
                    y1 = int(ymin * img_h)
                    x2 = int(xmax * img_w)
                    y2 = int(ymax * img_h)
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y2 = max(0, min(y2, img_h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    detections.append((x1, y1, x2, y2, cls_id, score))
        else:
            # Fallback: try to interpret as Nx6 or Nx7 like [x1,y1,x2,y2,score,cls(, ...)]
            # If your YOLOv5 no-NMS output is different, we keep old behavior (no preds) rather than guessing wrong.
            try:
                arr = np.asarray(nms_output)
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    # Heuristic: last column is class id, 5th is score
                    for row in arr:
                        x1, y1, x2, y2 = row[0:4]
                        score = float(row[4])
                        cls_id = int(row[5])
                        if score < conf_thres:
                            continue
                        x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))
                        x1 = max(0, min(x1, img_w - 1))
                        y1 = max(0, min(y1, img_h - 1))
                        x2 = max(0, min(x2, img_w - 1))
                        y2 = max(0, min(y2, img_h - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        detections.append((x1, y1, x2, y2, cls_id, score))
            except Exception:
                pass

    detections.sort(key=lambda x: x[5], reverse=True)
    if len(detections) > top_k:
        detections = detections[:top_k]
    return detections


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


def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float):
    """
    Standard greedy NMS.
    boxes_xyxy: (N,4) in pixels
    scores: (N,)
    returns indices kept
    """
    if boxes_xyxy.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes_xyxy[:, 0].astype(np.float32)
    y1 = boxes_xyxy[:, 1].astype(np.float32)
    x2 = boxes_xyxy[:, 2].astype(np.float32)
    y2 = boxes_xyxy[:, 3].astype(np.float32)

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union

        rest = rest[iou <= iou_thres]
        order = rest

    return np.asarray(keep, dtype=np.int64)


def apply_python_nms(detections, iou_thres=0.45, class_agnostic=False, max_det=200):
    """
    detections: list of (x1,y1,x2,y2,cls,score)
    returns filtered detections (same format), sorted by score desc.
    """
    if not detections:
        return detections

    dets = np.array(detections, dtype=np.float32)
    boxes = dets[:, 0:4]
    cls = dets[:, 4].astype(np.int32)
    scores = dets[:, 5]

    keep_all = []

    if class_agnostic:
        keep = nms_xyxy(boxes, scores, iou_thres)
        keep_all = keep.tolist()
    else:
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            keep_c = nms_xyxy(boxes[idx], scores[idx], iou_thres)
            keep_all.extend(idx[keep_c].tolist())

    # sort kept by score
    keep_all = sorted(keep_all, key=lambda i: float(scores[i]), reverse=True)
    if max_det and len(keep_all) > int(max_det):
        keep_all = keep_all[: int(max_det)]

    out = [detections[i] for i in keep_all]
    return out


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
            frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                continue
            resized_bgr = cv2.resize(frame_bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
            self.q.put((img_path, resized_bgr, rgb))
        self.q.put(None)


def list_images_sorted(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True, help="Path to HEF")
    ap.add_argument("--images", required=True, help="Images directory (sequence)")
    ap.add_argument("--labels", default=None, help="Labels directory (same stem as image), optional")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (pre-filter)")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold for TP matching (evaluation)")
    ap.add_argument("--topk", type=int, default=200, help="Top-K detections BEFORE Python NMS")
    ap.add_argument("--stride", type=int, default=1, help="Frame stride (2 means skip every other image)")
    ap.add_argument("--num-images", type=int, default=0, help="Limit number of images to process (0 = all)")
    ap.add_argument("--headless", type=int, default=0, help="1 disables imshow")
    ap.add_argument("--show-gt", type=int, default=1, help="1 draws GT boxes if labels provided")
    ap.add_argument("--show-pred", type=int, default=1, help="1 draws prediction boxes")

    # ===== New: Python-side NMS controls =====
    ap.add_argument("--nms", type=int, default=0, help="1 enables Python NMS, 0 disables (default keeps old behavior)")
    ap.add_argument("--nms-iou", type=float, default=0.45, help="IoU threshold for Python NMS")
    ap.add_argument("--nms-class-agnostic", type=int, default=0, help="1 = class-agnostic NMS, 0 = per-class NMS")
    ap.add_argument("--nms-max-det", type=int, default=200, help="Max detections after Python NMS")

    args = ap.parse_args()

    hef_path = args.hef
    images_dir = Path(args.images).expanduser().resolve()
    labels_dir = Path(args.labels).expanduser().resolve() if args.labels else None

    print("Using HEF:", hef_path)
    print("Images dir:", images_dir)
    print("Labels dir:", labels_dir if labels_dir else "(none)")
    print(f"Python NMS: {'ON' if args.nms else 'OFF'} | nms_iou={args.nms_iou} "
          f"| class_agnostic={bool(args.nms_class_agnostic)} | nms_max_det={args.nms_max_det}")

    all_images = list_images_sorted(images_dir)
    if not all_images:
        print("[ERROR] No images found.")
        return

    stride = max(1, int(args.stride))
    images = all_images[::stride]

    if args.num_images and args.num_images > 0:
        images = images[: args.num_images]

    print(f"Total images on disk: {len(all_images)} | after stride={stride}: {len(all_images[::stride])} | will process: {len(images)}")

    hef = hpf.HEF(hef_path)
    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe
        )
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstream_info = hef.get_input_vstream_infos()[0]
        output_vstream_infos = hef.get_output_vstream_infos()
        output_vstream_info = output_vstream_infos[0]

        in_h, in_w, in_c = input_vstream_info.shape

        print("Input shape from HEF:", input_vstream_info.shape)
        print("Input vstream name:", input_vstream_info.name)
        print("Output vstream name:", output_vstream_info.name)

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.UINT8
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )

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

        last_loop_end = time.time()
        est_npu_duty = 100.0

        fps_sum = 0.0
        fps_min = float("inf")
        fps_max = 0.0
        fps_count = 0

        infer_sum = 0.0
        infer_min = float("inf")
        infer_max = 0.0
        infer_count = 0

        duty_sum = 0.0
        duty_min = float("inf")
        duty_max = 0.0
        duty_count = 0

        cpu_sum = 0.0
        cpu_min = float("inf")
        cpu_max = 0.0
        cpu_count = 0

        ram_sum = 0.0
        ram_min = float("inf")
        ram_max = 0.0
        ram_count = 0

        # Eval accumulators
        cls_tp = np.zeros(len(SP14_NAMES), dtype=np.int64)
        cls_fp = np.zeros(len(SP14_NAMES), dtype=np.int64)
        cls_fn = np.zeros(len(SP14_NAMES), dtype=np.int64)
        cls_gt_count = np.zeros(len(SP14_NAMES), dtype=np.int64)
        cls_records = {c: [] for c in range(len(SP14_NAMES))}

        frame_id = 0
        first_frame = True

        if not args.headless:
            cv2.namedWindow("Hailo Dataset Stream", cv2.WINDOW_NORMAL)

        try:
            with network_group.activate(network_group_params):
                with hpf.InferVStreams(
                    network_group,
                    input_vstreams_params,
                    output_vstreams_params,
                    tf_nms_format=False,
                ) as infer_pipeline:

                    while True:
                        loop_start = time.time()

                        item = prefetch.q.get()
                        if item is None:
                            print("end (no more images)")
                            break

                        img_path, resized_bgr, rgb = item
                        frame_id += 1

                        input_data = {input_vstream_info.name: np.expand_dims(rgb, axis=0)}
                        infer_start = time.time()
                        results = infer_pipeline.infer(input_data)
                        infer_end = time.time()

                        infer_ms = (infer_end - infer_start) * 1000.0
                        infer_sum += infer_ms
                        infer_count += 1
                        infer_min = min(infer_min, infer_ms)
                        infer_max = max(infer_max, infer_ms)

                        loop_end = time.time()
                        loop_dt = loop_end - last_loop_end
                        last_loop_end = loop_end

                        if loop_dt > 0.0:
                            duty = (infer_end - infer_start) / loop_dt
                            duty = max(0.0, min(1.0, duty))
                            est_npu_duty = duty * 100.0
                            duty_sum += est_npu_duty
                            duty_count += 1
                            duty_min = min(duty_min, est_npu_duty)
                            duty_max = max(duty_max, est_npu_duty)

                        dt = loop_end - loop_start
                        if dt > 0.0:
                            fps_inst = 1.0 / dt
                            fps_window.append(fps_inst)
                            fps_smooth = sum(fps_window) / len(fps_window)
                            fps_sum += fps_inst
                            fps_count += 1
                            fps_min = min(fps_min, fps_inst)
                            fps_max = max(fps_max, fps_inst)

                        nms_output = results[output_vstream_info.name][0]
                        if first_frame:
                            first_frame = False
                            print("First frame output type:", type(nms_output))
                            if isinstance(nms_output, np.ndarray):
                                print("First frame output shape:", nms_output.shape, "dtype:", nms_output.dtype)

                        # 1) Decode to flat detections
                        preds = postprocess_hailo_nms(
                            nms_output, img_w=in_w, img_h=in_h, conf_thres=args.conf, top_k=args.topk
                        )

                        # 2) Optional Python-side NMS
                        if args.nms:
                            preds = apply_python_nms(
                                preds,
                                iou_thres=float(args.nms_iou),
                                class_agnostic=bool(args.nms_class_agnostic),
                                max_det=int(args.nms_max_det),
                            )

                        gts = []
                        if labels_dir is not None:
                            lbl_path = labels_dir / f"{img_path.stem}.txt"
                            gts = load_gt_yolo(lbl_path, img_w=in_w, img_h=in_h)

                            matched = match_per_class(preds, gts, iou_thres=args.iou)
                            for c, m in matched.items():
                                if 0 <= c < len(SP14_NAMES):
                                    cls_tp[c] += m["tp"]
                                    cls_fp[c] += m["fp"]
                                    cls_fn[c] += m["fn"]
                                    cls_gt_count[c] += m["gt_count"]
                                    cls_records[c].extend(m["records"])

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

                        if args.show_gt and gts:
                            for (x1, y1, x2, y2, cls) in gts:
                                if 0 <= cls < len(SP14_NAMES):
                                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(
                                        overlay,
                                        f"GT:{cls}:{SP14_NAMES[cls]}",
                                        (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45,
                                        (255, 0, 0),
                                        1,
                                        cv2.LINE_AA,
                                    )

                        if args.show_pred and preds:
                            for (x1, y1, x2, y2, cls_id, score) in preds:
                                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                name = SP14_NAMES[cls_id] if 0 <= cls_id < len(SP14_NAMES) else str(cls_id)
                                cv2.putText(
                                    overlay,
                                    f"P:{cls_id}:{name} {score:.2f}",
                                    (x1, max(0, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

                        lines = [
                            f"Frame: {frame_id} | {img_path.name}",
                            f"FPS: {fps_smooth:5.1f}",
                            f"Infer: {infer_ms:6.1f} ms",
                            f"NPU est duty: {est_npu_duty:5.1f}%",
                            f"CPU: {cpu_percent:5.1f}%",
                            f"RAM: {mem_mb:7.1f} MB",
                            f"conf={args.conf:.2f} eval_iou={args.iou:.2f} topk(pre)={args.topk} stride={stride} num_images={args.num_images or 'all'}",
                            f"py_nms={'ON' if args.nms else 'OFF'} nms_iou={args.nms_iou:.2f} agnostic={bool(args.nms_class_agnostic)} max_det={args.nms_max_det}",
                        ]

                        if labels_dir is not None:
                            tp_all = int(cls_tp.sum())
                            fp_all = int(cls_fp.sum())
                            fn_all = int(cls_fn.sum())
                            prec_all = tp_all / (tp_all + fp_all + 1e-9)
                            rec_all = tp_all / (tp_all + fn_all + 1e-9)
                            f1_all = (2 * prec_all * rec_all) / (prec_all + rec_all + 1e-9)
                            lines.append(f"Eval so far P/R/F1: {prec_all:.3f}/{rec_all:.3f}/{f1_all:.3f}")

                        panel_w = 860
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

                        cv2.imshow("Hailo Dataset Stream", overlay)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break

        finally:
            prefetch.stop()
            if not args.headless:
                cv2.destroyAllWindows()

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

        if duty_count > 0:
            duty_avg = duty_sum / duty_count
            print(f"NPU est duty (%): avg={duty_avg:.2f}, min={duty_min:.2f}, max={duty_max:.2f}")
        else:
            print("NPU est duty (%): no data")

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
            print("\n========== Quantitative evaluation (IoU>=%.2f) ==========" % args.iou)

            ap_list = []
            valid_ap_classes = 0

            for c in range(len(SP14_NAMES)):
                tp = int(cls_tp[c])
                fp = int(cls_fp[c])
                fn = int(cls_fn[c])
                gt = int(cls_gt_count[c])

                if gt == 0 and (tp + fp) == 0:
                    continue

                prec = tp / (tp + fp + 1e-9)
                rec = tp / (tp + fn + 1e-9)
                f1 = (2 * prec * rec) / (prec + rec + 1e-9)

                apv = ap_from_records(cls_records[c], gt)
                ap_str = "nan" if np.isnan(apv) else f"{apv:.3f}"

                print(f"[{c:2d}] {SP14_NAMES[c]:13s} | GT={gt:5d} TP={tp:5d} FP={fp:5d} FN={fn:5d} "
                      f"| P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f} | AP50={ap_str}")

                if not np.isnan(apv):
                    ap_list.append(apv)
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
