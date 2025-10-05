# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 22:24:05 2025

@author: Milan
"""

import argparse
import cv2
from tracker import DroneTrackerApp
from multi_tracker import MultiObjectTracker
from ultralytics import YOLO


def run_opencv(args):
    app = DroneTrackerApp(args)
    app.run()


def run_yolo(args):
    # načteme YOLOv8 model (univerzální n)
    model = YOLO("yolov8n.pt")
    tracker = MultiObjectTracker(winsize=args.winsize)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"❌ Nelze otevřít zdroj: {args.source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detekce
        results = model(frame, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update(detections, frame)

        # vykreslení
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = t.to_ltrb()
            cv2.rectangle(frame, (int(l), int(t_)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {t.track_id}", (int(l), int(t_) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.show:
            tracker.show(frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["opencv", "yolo"], default="opencv", help="Režim trackeru")
    parser.add_argument("--source", required=True, help="Video nebo kamera")
    parser.add_argument("--output", default=None, help="Cílový video soubor")
    parser.add_argument("--show", action="store_true", help="Zobrazit okno s výsledkem")
    parser.add_argument("--pause_frame", type=int, default=None, help="Frame, kde zastavit a vybrat ROI (jen opencv mód)")
    parser.add_argument("--winsize", type=int, default=None, help="Šířka okna pro zobrazení")
    parser.add_argument("--tracker", default="KCF", help="Typ trackeru pro OpenCV mód (např. KCF, CSRT, MOSSE)")

    args = parser.parse_args()

    if args.mode == "opencv":
        run_opencv(args)
    elif args.mode == "yolo":
        run_yolo(args)
