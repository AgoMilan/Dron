# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 22:34:42 2025

@author: Milan
"""

# yolo11_face_tracker_v2_8.py
# -*- coding: utf-8 -*-
"""
YOLO11 + DeepSORT + FaceRecognitionManager + VisitorDB v2
- automatická evidence návštěvníků
- zamezení duplicitních návštěv během jedné přítomnosti (revisit_timeout)
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from visitor_db_v2 import VisitorDB
from face_recognition_manager_v2_7 import FaceRecognitionManager


def extract_face_detections(res, frame_w, frame_h):
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()
    img_h, img_w = res.orig_shape
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4]
        conf = float(confs[i])
        name = res.names.get(int(clss[i]), str(int(clss[i]))) if hasattr(res, "names") else "face"
        if "face" not in name.lower():
            continue
        # přepočet
        x1 *= frame_w / img_w
        x2 *= frame_w / img_w
        y1 *= frame_h / img_h
        y2 *= frame_h / img_h
        dets.append(([x1, y1, x2, y2], conf, "face"))
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rtsp://root:heslo@192.168.0.205/live.sdp")
    parser.add_argument("--model", type=str, default="YOLO11face.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--shrink", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gallery", type=str, default="gallery.json")
    parser.add_argument("--visitors", type=str, default="visitors.json")
    parser.add_argument("--auto_add_threshold", type=int, default=2)
    parser.add_argument("--revisit_timeout", type=int, default=900, help="čas v sekundách pro opakovanou návštěvu (default 15 min)")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    print(f"[MAIN] Device: {device}")
    print(f"[MAIN] Loading model {args.model}")
    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        print("[MAIN] Nelze otevřít zdroj.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[MAIN] Kamera {frame_w}x{frame_h} @ {fps:.1f}FPS")

    visitor_db = VisitorDB(path=args.visitors, threshold=0.6, revisit_timeout=args.revisit_timeout)
    frm = FaceRecognitionManager(
        device=device,
        gallery_path=args.gallery,
        decision_min_samples=3,
        threshold=0.65,
        visitor_db=visitor_db,
        auto_add_threshold=args.auto_add_threshold,
    )

    tracker = DeepSort(max_age=3, n_init=1, max_iou_distance=0.6)

    if args.display:
        cv2.namedWindow("Tracker v2.8", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracker v2.8", int(frame_w * 0.7), int(frame_h * 0.7))

    print("[MAIN] Spuštěno. Klávesy: E=enroll, D=delete, G=gallery, V=visitors, R=rename, Q/Esc=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
            res0 = results[0] if results else None
            dets = extract_face_detections(res0, frame_w, frame_h)
            tracks = tracker.update_tracks(dets, frame=frame)
            display = frame.copy()

            for tr in tracks:
                if not tr.is_confirmed() or tr.time_since_update > 1:
                    continue

                x1, y1, x2, y2 = map(float, tr.to_tlbr())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                side = min((x2 - x1), (y2 - y1))
                vs = side * args.shrink
                vx1, vy1, vx2, vy2 = cx - vs / 2, cy - vs / 2, cx + vs / 2, cy + vs / 2
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(frame_w - 1, vx2), min(frame_h - 1, vy2)

                label, score, vinfo = frm.update_track(frame, tr.track_id, (x1, y1, x2, y2))
                color = (0, 255, 0)
                if label.startswith("unknown"):
                    color = (0, 140, 255)
                elif vinfo is not None:
                    color = (0, 255, 255)
                cv2.rectangle(display, (int(vx1), int(vy1)), (int(vx2), int(vy2)), color, 2)
                cv2.putText(display, f"{label}", (int(vx1), int(vy1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if args.display:
                cv2.imshow("Tracker v2.8", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        print("[MAIN] Ukončeno uživatelem.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[MAIN] Konec.")


if __name__ == "__main__":
    main()
