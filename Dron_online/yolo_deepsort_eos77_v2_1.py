# -*- coding: utf-8 -*-
"""
yolo_deepsort_eos77_v2_1.py
---------------------------------
Optimized YOLOv8 + DeepSORT live capture for Canon EOS 77D (USB/index 0)
- maps YOLO detections to original camera resolution
- shrink option (centered, no shift)
- sets camera capture resolution to reduce latency (1280x720)
- faster tracker defaults (n_init=1, max_age=20)
- robust handling of None confidences
Save as: yolo_deepsort_eos77_v2_1.py
Run example:
    python yolo_deepsort_eos77_v2_1.py --source 0 --display --winsize 1000 --shrink 0.3
"""

import argparse
import time
import os
import csv
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def extract_detections_from_result(res, frame_w, frame_h):
    """
    Convert YOLO result boxes (which may be relative to a resized inference image)
    back to the original frame resolution (frame_w x frame_h).
    Returns list of ([x1,y1,x2,y2], conf, class_name)
    """
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    # Attempt to read original inference shape; fallback to res.ori... if not present
    try:
        img_h, img_w = res.orig_shape[0], res.orig_shape[1]
    except Exception:
        # If not available, assume YOLO used same size as frame (no scaling)
        img_h, img_w = frame_h, frame_w

    # scale factors to map detection coords back to frame
    scale_x = frame_w / img_w if img_w != 0 else 1.0
    scale_y = frame_h / img_h if img_h != 0 else 1.0

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        # map to original frame size
        x1_m = x1 * scale_x
        x2_m = x2 * scale_x
        y1_m = y1 * scale_y
        y2_m = y2 * scale_y
        conf = float(confs[i]) if confs[i] is not None else 0.0
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        dets.append(([x1_m, y1_m, x2_m, y2_m], conf, cls_name))
    return dets

def init_video_writer(path, fourcc, fps, size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, fourcc, fps, size)

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT for EOS77D (optimized)")
    parser.add_argument("--source", type=str, required=True, help="camera index (0) or RTSP URL")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="ultralytics model")
    parser.add_argument("--conf", type=float, default=0.35, help="yolo confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="yolo inference size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--display", action="store_true", help="show window")
    parser.add_argument("--winsize", type=int, default=1000, help="display width in px (keeps aspect)")
    parser.add_argument("--shrink", type=float, default=0.3, help="shrink bounding box (0.3 = 30%)")
    parser.add_argument("--output", type=str, default="runs/eos_out.mp4", help="optional output video")
    parser.add_argument("--csv", type=str, default="runs/eos_log.csv", help="optional csv log")
    parser.add_argument("--step", type=int, default=1, help="detect every N frames")
    parser.add_argument("--max_age", type=int, default=20, help="DeepSort max_age (frames)")
    parser.add_argument("--n_init", type=int, default=1, help="DeepSort n_init (frames until confirmed)")
    args = parser.parse_args()

    # determine source index or url
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("Loading model:", args.model)
    model = YOLO(args.model)

    print("Opening camera:", source)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # use DirectShow on Windows to help stable capture
    # Try to set camera capture resolution to reduce latency (reduce data over USB)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    # read frame properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera opened: {frame_w}x{frame_h} @ {fps:.2f}fps")

    # prepare writer and csv
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = init_video_writer(args.output, fourcc, fps, (frame_w, frame_h))
        print("Output video:", args.output)

    csv_f = None
    csv_writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_f = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["frame","time_s","track_id","class","conf","x1","y1","x2","y2"])
        print("CSV log:", args.csv)

    # initialize tracker with responsive settings
    tracker = DeepSort(max_age=args.max_age, n_init=args.n_init, embedder="mobilenet")

    # display window
    if args.display:
        cv2.namedWindow("EOS77D Live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("EOS77D Live", args.winsize, int(args.winsize * frame_h / frame_w))

    times = deque(maxlen=30)
    frame_idx = 0
    t0 = time.time()

    print("▶️ Starting processing loop. Press Q or ESC to quit.")

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed, exiting.")
            break
        frame_idx += 1

        # run detection every N frames (args.step)
        dets = []
        if frame_idx % args.step == 0:
            # YOLO returns results; we map boxes to original frame size using helper
            results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf,
                                    device=args.device, verbose=False)
            if results and len(results) > 0:
                dets = extract_detections_from_result(results[0], frame_w, frame_h)

        # update tracker with detections in original frame coordinates
        try:
            tracks = tracker.update_tracks(dets, frame=frame)
        except Exception as e:
            print("Tracker error:", e)
            tracks = []

        display = frame.copy()
        scale = args.winsize / frame_w if args.winsize and frame_w > 0 else 1.0
        det_count = 0

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            # Use float coords from tracker
            x1, y1, x2, y2 = map(float, tr.to_tlbr())
            # compute precise center as float
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # shrink symmetrically around the float center, then round for drawing
            shrink = args.shrink if args.shrink > 0.0 else 1.0
            if shrink < 1.0:
                bw = (x2 - x1) * shrink
                bh = (y2 - y1) * shrink
                x1_s = cx - bw / 2.0
                x2_s = cx + bw / 2.0
                y1_s = cy - bh / 2.0
                y2_s = cy + bh / 2.0
            else:
                x1_s, y1_s, x2_s, y2_s = x1, y1, x2, y2

            # clamp to image boundaries
            x1_s = max(0.0, x1_s)
            y1_s = max(0.0, y1_s)
            x2_s = min(frame_w - 1.0, x2_s)
            y2_s = min(frame_h - 1.0, y2_s)

            # safe class/conf extraction
            cls = getattr(tr, "det_class", "?")
            conf = getattr(tr, "det_conf", 0.0) or 0.0
            label = f"ID:{tid} {cls} {conf:.2f}"

            # draw using rounded coordinates mapped to display scale
            p1 = (int(round(x1_s * scale)), int(round(y1_s * scale)))
            p2 = (int(round(x2_s * scale)), int(round(y2_s * scale)))
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
            cv2.putText(display, label, (p1[0], max(15, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # optional center marker (small cross)
            cv2.drawMarker(display, (int(round(cx * scale)), int(round(cy * scale))),
                           (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)

            det_count += 1

            # write csv (original coordinates)
            if csv_writer:
                csv_writer.writerow([frame_idx, time.time() - t0, tid, cls, conf,
                                     int(round(x1_s)), int(round(y1_s)), int(round(x2_s)), int(round(y2_s))])

        # write output video (original frame)
        if writer:
            writer.write(frame)

        # show fps and object count
        t_end = time.time()
        times.append(t_end - t_start)
        fps_sm = len(times) / sum(times) if sum(times) > 0 else 0.0
        cv2.putText(display, f"FPS: {fps_sm:.1f} Obj:{det_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.display:
            cv2.imshow("EOS77D Live", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            print("Exit requested by user.")
            break

    cap.release()
    if writer:
        writer.release()
    if csv_f:
        csv_f.close()
    if args.display:
        cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"Done — frames: {frame_idx}, time: {dt:.1f}s, avg FPS: {frame_idx/dt:.2f}")

if __name__ == "__main__":
    main()
