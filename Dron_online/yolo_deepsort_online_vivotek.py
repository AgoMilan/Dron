# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:24:19 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
yolo_deepsort_online_vivotek.py
Verze pro připojení na Vivotek IP kameru (RTSP) + YOLOv8 + DeepSORT
- fallback na lokální webkameru (index 0), pokud RTSP selže
- jednoduché PTZ ovládání přes HTTP GET (configurable)
- spouštění:
    python yolo_deepsort_online_vivotek.py --source "rtsp://root:asd456@192.168.0.205:554/live.sdp" --display

Požadavky (pip):
    pip install ultralytics opencv-python-headless deep-sort-realtime numpy pandas requests
    (pokud chceš grafické okno použij opencv-python místo headless)

Autor: Milan + GPT úpravy
"""
import argparse
import time
import os
import sys
import csv
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import pandas as pd
import requests

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -------------------------
# PTZ helpers (HTTP)
# -------------------------
class PTZController:
    """
    Jednoduchý PTZ ovladač pomocí HTTP GET (Vivotek / generické CGI endpointy).
    ptz_url_template příklad:
      "http://root:asd456@192.168.0.205/cgi-bin/camctrl/ptz.cgi?move={cmd}"
    kde {cmd} je one of: left,right,up,down,stop
    """
    def __init__(self, ptz_url_template=None, timeout=1.0, cooldown=0.5):
        self.ptz_url_template = ptz_url_template
        self.timeout = timeout
        self.cooldown = cooldown
        self._last_cmd_time = 0.0

    def enabled(self):
        return bool(self.ptz_url_template)

    def send(self, cmd):
        if not self.enabled():
            print("PTZ: not enabled (no URL template).")
            return False

        now = time.time()
        if now - self._last_cmd_time < self.cooldown:
            # cooldown to avoid spamming camera
            return False
        self._last_cmd_time = now

        url = self.ptz_url_template.format(cmd=cmd)
        try:
            r = requests.get(url, timeout=self.timeout)
            if r.status_code in (200, 201, 204):
                print(f"PTZ: sent '{cmd}' (OK)")
                return True
            else:
                print(f"PTZ: sent '{cmd}' -> HTTP {r.status_code}")
                return False
        except Exception as e:
            print(f"PTZ: error sending '{cmd}': {e}")
            return False

    # convenience methods
    def left(self): return self.send("left")
    def right(self): return self.send("right")
    def up(self): return self.send("up")
    def down(self): return self.send("down")
    def stop(self): return self.send("stop")

# -------------------------
# Utility functions
# -------------------------
def try_open_rtsp(rtsp_url, test_frames=3, wait_sec=0.5):
    """
    Zkus otevřít RTSP a přečíst několik rámců.
    Vrací cap nebo None.
    """
    print("Ověřuji RTSP:", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("RTSP: nelze otevřít stream.")
        try:
            cap.release()
        except:
            pass
        return None

    ok = False
    for i in range(test_frames):
        ret, _ = cap.read()
        if ret:
            ok = True
            break
        time.sleep(wait_sec)
    if not ok:
        print("RTSP: otevřeno, ale žádné obrázky (timeout). Zavírám.")
        cap.release()
        return None
    print("RTSP: stream OK.")
    # resetněme pozici není třeba pro RTSP
    return cap

def init_video_writer(path, fourcc, fps, size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return cv2.VideoWriter(path, fourcc, fps, size)

def extract_detections_from_result(res):
    """
    Převod výsledku ultralytics na formát DeepSort očekává:
    list of ([x1,y1,x2,y2], conf, class_name)
    """
    dets = []
    if res is None:
        return dets
    boxes = getattr(res, "boxes", None)
    if boxes is None:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, b[:4])
        conf = float(confs[i])
        cls_id = int(clss[i])
        class_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        dets.append(([x1, y1, x2, y2], conf, class_name))
    return dets

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO + DeepSORT online pro Vivotek (RTSP) s PTZ ovládáním")
    parser.add_argument("--source", type=str, required=True,
                        help="RTSP URL nebo číslo kamery (0) nebo cesta k souboru")
    parser.add_argument("--output", type=str, default="runs/vivotek_out.mp4", help="výstupní video (volitelné)")
    parser.add_argument("--csv", type=str, default="runs/vivotek_log.csv", help="CSV log (volitelné)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="ultralytics model")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu nebo cuda:0")
    parser.add_argument("--display", action="store_true", help="zobrazovat okno s výsledkem")
    parser.add_argument("--winsize", type=int, default=640, help="šířka okna při zobrazení (0=no resize)")
    parser.add_argument("--step", type=int, default=1, help="detekce každých N snímků")
    parser.add_argument("--max_age", type=int, default=30, help="DeepSort max_age")
    parser.add_argument("--n_init", type=int, default=1, help="DeepSort n_init")
    parser.add_argument("--ptz_url", type=str, default=None,
                        help="PTZ URL template, např. 'http://root:asd456@192.168.0.205/cgi-bin/camctrl/ptz.cgi?move={cmd}'")
    parser.add_argument("--ptz_cooldown", type=float, default=0.5, help="cooldown mezi PTZ prikazy (s)")
    args = parser.parse_args()

    # Inicializace modelu
    print("Načítám model:", args.model)
    model = YOLO(args.model)

    # PTZ controller (pokud zadan)
    ptz = PTZController(ptz_url_template=args.ptz_url, cooldown=args.ptz_cooldown) if args.ptz_url else PTZController(None)

    # Zkus RTSP nebo fallback
    cap = None
    # pokud je zdroj číslo (např. "0"), nech OpenCV použít lokální kameru
    if args.source.isdigit():
        src = int(args.source)
        print("Používám lokální kameru index:", src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Nelze otevřít lokální kameru: {src}")
    else:
        # pokusíme RTSP přes helper
        cap = try_open_rtsp(args.source)
        if cap is None:
            # fallback na lokální kameru 0
            print("Fallback: pokouším se připojit na lokální kameru 0")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Nelze otevřít RTSP ani lokální kameru (0). Zkontroluj připojení.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    print(f"Zdroj velikost: {w}x{h} @ {fps}fps")

    # Video writer
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = init_video_writer(args.output, fourcc, fps, (w, h))

    # CSV logger
    csv_f = None
    csv_writer = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_f = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["frame", "time_s", "track_id", "class", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h"])

    # Inicializace DeepSort trackeru
    tracker = DeepSort(max_age=args.max_age, n_init=args.n_init, embedder="mobilenet")

    frame_idx = 0
    t0 = time.time()
    pause_for_roi = False

    # Display window
    if args.display:
        cv2.namedWindow("Dron - Vivotek", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dron - Vivotek", 640, 480)  # pevná velikost okna

    # small deque for fps smoothing
    times = deque(maxlen=30)

    print("▶️ Spouštím smyčku zpracování. Klávesy: Q nebo ESC ukončí. PTZ: ←/a, →/d, ↑/w, ↓/s, p=stop")
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❗ žádný frame (stream skončil). Končím.")
            break
        frame_idx += 1

        # optionally resize for speed/display
        orig_h, orig_w = frame.shape[:2]
        display_frame = frame.copy()
        if args.winsize and orig_w > args.winsize:
            scale = args.winsize / orig_w
            display_frame = cv2.resize(display_frame, (int(orig_w * scale), int(orig_h * scale)))
        else:
            scale = 1.0

        # detection every N frames
        dets = []
        if frame_idx % args.step == 0:
            results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
            if results and len(results) > 0:
                r = results[0]
                # store names in r.names (ultralytics)
                dets = extract_detections_from_result(r)

        # update tracker
        try:
            tracks = tracker.update_tracks(dets, frame=frame)
        except Exception as e:
            print("Tracker update error:", e)
            tracks = []

        # draw tracks
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            tlbr = tr.to_tlbr()
            x1, y1, x2, y2 = map(int, tlbr)
            w_box = x2 - x1
            h_box = y2 - y1
            cx = x1 + w_box // 2
            cy = y1 + h_box // 2
            cls = getattr(tr, "det_class", "") or (tr.get_det_class() if hasattr(tr, "get_det_class") else "")
            conf = getattr(tr, "det_conf", 0.0)
            # draw on display_frame scaled
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(display_frame, p1, p2, (0, 255, 0), 2)
            label = f"ID:{tid} {cls} {conf:.2f}"
            cv2.putText(display_frame, label, (p1[0], max(15, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # write CSV
            if csv_writer:
                csv_writer.writerow([frame_idx, time.time() - t0, tid, cls, conf, x1, y1, x2, y2, cx, cy, w_box, h_box])

        # show FPS
        t_end = time.time()
        times.append(t_end - t_start)
        fps_sm = len(times) / sum(times) if sum(times) > 0 else 0.0
        cv2.putText(display_frame, f"FPS: {fps_sm:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # display
        if args.display:
            cv2.imshow("Dron - Vivotek", display_frame)

        # write raw (originální velikost) to out
        if out is not None:
            out.write(frame)

        # key handling (supports arrow keys + wasd)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # arrow keys have special codes in some environments (81,82,83,84)
            if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                print("Ukončuji (klávesa).")
                break
            elif key in (81, ord('a')):  # left
                if ptz.enabled(): ptz.left()
            elif key in (83, ord('d')):  # right
                if ptz.enabled(): ptz.right()
            elif key in (82, ord('w')):  # up
                if ptz.enabled(): ptz.up()
            elif key in (84, ord('s')):  # down
                if ptz.enabled(): ptz.down()
            elif key in (ord('p'), ord('P')):  # stop
                if ptz.enabled(): ptz.stop()
            # other useful shortcuts can be added here

    # cleanup
    cap.release()
    if out is not None:
        out.release()
    if csv_f:
        csv_f.close()
    if args.display:
        cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"✅ Hotovo — frames: {frame_idx}, time: {dt:.1f}s, avg FPS: {frame_idx/dt:.2f}")

if __name__ == "__main__":
    main()
