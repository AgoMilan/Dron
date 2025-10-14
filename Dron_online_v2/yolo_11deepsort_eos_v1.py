# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 17:02:07 2025

@author: Milan


Created on Sun Oct 12 2025
@author: Milan

YOLOv11 + DeepSORT
Live stream z Canon EOS 77D (USB / index 0)
--------------------------------------------
Spuštění:
    python yolo_deepsort_eos77_v11.py --source 0 --model yolov11n.pt --conf 0.35 --imgsz 640 --display --winsize 900 --shrink 0.3
"""

import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def extract_detections_from_result(res, frame_w, frame_h):
    """Vrátí detekce přepočítané na původní velikost videa."""
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    # Nové API YOLOv11 je kompatibilní s v8 – zůstává stejné
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
    parser.add_argument("--model", type=str, default="yolov11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--winsize", type=int, default=1280)
    parser.add_argument("--shrink", type=float, default=0.5)
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    print(f"Načítám model: {args.model}")
    model = YOLO(args.model)  # YOLOv11 automaticky rozpoznán

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Nelze otevřít kameru.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Kamera otevřena {w}x{h}@{fps:.1f} FPS")

    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet")

    cv2.namedWindow("EOS77D Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EOS77D Live", args.winsize, int(args.winsize * h / w))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv11 predikce (stejné API jako YOLOv8)
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False)
        dets = extract_detections_from_result(results[0], w, h) if results else []

        tracks = tracker.update_tracks(dets, frame=frame)
        display = frame.copy()
        scale = args.winsize / w

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            x1, y1, x2, y2 = map(float, tr.to_tlbr())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Shrink kolem středu
            shrink = args.shrink
            bw = (x2 - x1) * shrink
            bh = (y2 - y1) * shrink
            x1, x2 = cx - bw / 2, cx + bw / 2
            y1, y2 = cy - bh / 2, cy + bh / 2

            # Ořez podle hran
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

            # Zaměřovač do středu
            cv2.drawMarker(display, (int(cx * scale), int(cy * scale)), (0, 255, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

        cv2.imshow("EOS77D Live", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
