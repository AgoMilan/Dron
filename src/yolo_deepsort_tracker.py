# -*- coding: utf-8 -*-
"""
yolo_deepsort_tracker.py
Pokročilý tracking: YOLO detektor + DeepSORT
Autor: generováno ChatGPT (Milan project)

Požadavky (pip):
  pip install ultralytics opencv-python-headless deep-sort-realtime numpy pandas tqdm

Poznámky:
- Skript detekuje objekty pomocí YOLO (ultralytics) a následně je sleduje pomocí DeepSORT.
- Hledá objekty kategorií 'bird', 'drone' a obecně 'person' pokud chcete testovat.
- Pokud nemáte GPU, použijte --device cpu.
- Skript ukládá výstupní video a CSV log s tracky.
"""

import argparse
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO  # ultralytics (YOLOv8)            
from deep_sort_realtime.deepsort_tracker import DeepSort  # deep-sort-realtime

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2
    cy = y1 + h/2
    return (int(cx), int(cy), int(w), int(h))

def init_video_writer(path, fourcc, fps, size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, fourcc, fps, size)

def main(args):
    # Load model
    print("Loading YOLO model:", args.model)
    model = YOLO(args.model)  # supports 'yolov8n.pt' or 'yolov8n' (auto)
    # device selection via model.predict args below (ultralytics uses auto if not specified)

    # Initialize DeepSort
    print("Inicializuji DeepSort...")
    tracker = DeepSort(max_age=args.max_age,  # frames to keep unmatched tracks
                       n_init=args.n_init,     # frames to confirm track
                       max_cosine_distance=0.2,
                       nn_budget=100,
                       embedder="mobilenet")   # embedder name supported by package

    # Open source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Nelze otevřít zdroj: " + str(args.source))

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Zdroj: {args.source}  {w}x{h} @ {fps}fps")

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = init_video_writer(args.output, fourcc, fps, (w, h))

    # CSV logger
    logs = []
    header = ["frame", "time", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h"]

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        # optionally resize for speed
        if args.winsize and frame.shape[1] > args.winsize:
            scale = args.winsize / frame.shape[1]
            frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        # Run detection every N frames (args.step)
        dets = []
        if frame_idx % args.step == 0:
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            # ultralytics returns list of results (one per image)
            if len(results) > 0:
                r = results[0]
                boxes = getattr(r, "boxes", [])
                for box in boxes:
                    # each box: .xyxy .conf .cls
                    xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
                    x1,y1,x2,y2 = map(int, xyxy.tolist())
                    conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, "cpu") else float(box.conf)
                    cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, "cpu") else int(box.cls)
                    class_name = model.names.get(cls, str(cls))
                    if args.classes and class_name not in args.classes:
                        continue
                    dets.append(([x1,y1,x2,y2], conf, class_name))

        # Update tracker with detections
        tracks = tracker.update_tracks(dets, frame=frame)  # list of Track objects
        # tracks: has .track_id, .to_tlbr(), .det_conf, .get_det_class()
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            tlbr = tr.to_tlbr()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, tlbr)
            w_box = x2 - x1
            h_box = y2 - y1
            cx = x1 + w_box//2
            cy = y1 + h_box//2
            cls = getattr(tr, "det_class", "") or tr.get_det_class() if hasattr(tr, "get_det_class") else ""
            conf = getattr(tr, "det_conf", 0.0)
            # draw
            label = f"ID:{tid} {cls} {conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # log
            logs.append([frame_idx, time.time()-start_time, tid, cls, conf, x1, y1, x2, y2, cx, cy, w_box, h_box])

        # If no tracks, optionally mark "No targets"
        if args.display and len(tracks) == 0:
            cv2.putText(frame, "No tracks", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Write output
        if out is not None:
            out.write(frame)
        if args.display:
            cv2.imshow("YOLO + DeepSORT", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Save CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df = pd.DataFrame(logs, columns=header)
        df.to_csv(args.csv, index=False)
        print("CSV uložen:", args.csv)

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Hotovo. Výstup uložen do:", args.output if args.output else "nebyl zadán")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT tracker")
    parser.add_argument("--source", type=str, default=0, help="video file or camera index")
    parser.add_argument("--output", type=str, default="/mnt/data/runs/yolo_deepsort_out.mp4", help="výstupní video")
    parser.add_argument("--csv", type=str, default="/mnt/data/runs/yolo_deepsort_log.csv", help="CSV log")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="ultralytics model (yolov8n.pt) or path")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--step", type=int, default=1, help="run detection every N frames (1 => every)")
    parser.add_argument("--winsize", type=int, default=0, help="max width to scale for speed (0 no resize)")
    parser.add_argument("--fps", type=float, default=0, help="force fps if source has none")
    parser.add_argument("--display", action='store_true', help="show window")
    parser.add_argument("--classes", type=str, nargs='*', default=["bird","drone","person"], help="class names to keep")
    parser.add_argument("--max_age", type=int, default=30, help="DeepSort max_age")
    parser.add_argument("--n_init", type=int, default=1, help="frames to confirm track")
    args = parser.parse_args()
    main(args)
