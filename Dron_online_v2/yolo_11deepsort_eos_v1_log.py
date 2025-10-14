# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:04:08 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
YOLOv11 + DeepSORT + EOS77D (USB kamera)
----------------------------------------
Zobrazuje FPS, latenci a ukládá statistiky do CSV logu.
Spuštění:
    python yolo_11deepsort_eos_v3_fps_log.py --source 0 --model yolo11n.pt --conf 0.35 --imgsz 640 --display --winsize 900 --shrink 0.3
"""

import argparse
import time
import cv2
import csv
import os
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def extract_detections_from_result(res, frame_w, frame_h):
    """Vrátí detekce přepočítané na původní velikost videa."""
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    img_w, img_h = res.orig_shape[1], res.orig_shape[0]
    scale_x = frame_w / img_w
    scale_y = frame_h / img_h

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        conf = float(confs[i]) if confs[i] is not None else 0.0
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        dets.append(([x1, y1, x2, y2], conf, cls_name))
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--winsize", type=int, default=1280)
    parser.add_argument("--shrink", type=float, default=0.5)
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    # detekce CUDA / CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Načítám model: {args.model} ({device.upper()})")

    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Nelze otevřít kameru.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Kamera otevřena {w}x{h}@{fps_cam:.1f} FPS")

    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet")

    cv2.namedWindow("EOS77D Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EOS77D Live", args.winsize, int(args.winsize * h / w))

    # CSV logování
    log_file = "performance_log.csv"
    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, mode="a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["timestamp", "device", "fps", "latency_ms", "objects_detected"])

    frame_count = 0
    fps_display = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
        dets = extract_detections_from_result(results[0], w, h) if results else []
        tracks = tracker.update_tracks(dets, frame=frame)

        display = frame.copy()
        scale = args.winsize / w

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = map(float, tr.to_tlbr())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            shrink = args.shrink
            bw = (x2 - x1) * shrink
            bh = (y2 - y1) * shrink
            x1, x2 = cx - bw / 2, cx + bw / 2
            y1, y2 = cy - bh / 2, cy + bh / 2
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cls = getattr(tr, "det_class", "?")
            conf = getattr(tr, "det_conf", 0.0) or 0.0
            label = f"ID:{tr.track_id} {cls} {conf:.2f}"

            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
            cv2.putText(display, label, (p1[0], max(20, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.drawMarker(display, (int(cx * scale), int(cy * scale)), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        # FPS a latence
        frame_count += 1
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            last_time = time.time()

        latency_ms = (time.time() - start_time) * 1000
        overlay = f"{device.upper()} | FPS: {fps_display:.1f} | Latence: {latency_ms:.1f} ms | Objekty: {len(tracks)}"
        cv2.putText(display, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # log do CSV
        csv_writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            device,
            round(fps_display, 2),
            round(latency_ms, 2),
            len(tracks)
        ])
        csv_file.flush()

        cv2.imshow("EOS77D Live", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"✅ Log uložen: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    main()
