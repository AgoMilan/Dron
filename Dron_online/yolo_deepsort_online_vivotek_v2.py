# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:26:37 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
yolo_deepsort_online_vivotek_v2.py
----------------------------------
Verze pro připojení na Vivotek IP kameru (RTSP) + YOLOv8 + DeepSORT
- bezpečné ošetření NoneType (chyba při formátování)
- zobrazuje FPS a počet detekcí
- pevná velikost okna 640x480
- jednoduché PTZ ovládání přes HTTP GET (volitelné)
"""

import argparse
import time
import os
import csv
from collections import deque
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ===============================
# PTZ Controller
# ===============================
class PTZController:
    def __init__(self, ptz_url_template=None, timeout=1.0, cooldown=0.5):
        self.ptz_url_template = ptz_url_template
        self.timeout = timeout
        self.cooldown = cooldown
        self._last_cmd_time = 0.0

    def enabled(self):
        return bool(self.ptz_url_template)

    def send(self, cmd):
        if not self.enabled():
            return False

        now = time.time()
        if now - self._last_cmd_time < self.cooldown:
            return False

        self._last_cmd_time = now
        url = self.ptz_url_template.format(cmd=cmd)
        try:
            r = requests.get(url, timeout=self.timeout)
            print(f"PTZ: {cmd} -> {r.status_code}")
            return r.status_code in (200, 204)
        except Exception as e:
            print(f"PTZ error: {e}")
            return False

    def left(self): self.send("left")
    def right(self): self.send("right")
    def up(self): self.send("up")
    def down(self): self.send("down")
    def stop(self): self.send("stop")


# ===============================
# Helper Functions
# ===============================
def try_open_rtsp(rtsp_url, test_frames=3):
    print("Ověřuji RTSP:", rtsp_url)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ Nelze otevřít RTSP stream.")
        return None

    for _ in range(test_frames):
        ret, _ = cap.read()
        if ret:
            print("✅ RTSP: stream OK.")
            return cap
        time.sleep(0.5)

    print("⚠️ RTSP otevřen, ale žádné snímky.")
    cap.release()
    return None


def extract_detections_from_result(res):
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, b[:4])
        conf = float(confs[i])
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        dets.append(([x1, y1, x2, y2], conf, cls_name))
    return dets


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="YOLO + DeepSORT online pro Vivotek RTSP")
    parser.add_argument("--source", type=str, required=True, help="RTSP URL nebo index kamery (0)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--display", action="store_true", help="Zobrazit výsledek")
    parser.add_argument("--ptz_url", type=str, default=None,
                        help="např. http://root:asd456@192.168.0.205/cgi-bin/camctrl/ptz.cgi?move={cmd}")
    args = parser.parse_args()

    print("Načítám model:", args.model)
    model = YOLO(args.model)

    # PTZ
    ptz = PTZController(ptz_url_template=args.ptz_url) if args.ptz_url else PTZController(None)

    # Kamera
    cap = None
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = try_open_rtsp(args.source)
    if not cap or not cap.isOpened():
        raise RuntimeError("❌ Nelze otevřít zdroj videa.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Zdroj: {w}x{h} @ {fps:.1f} fps")

    # Tracker
    tracker = DeepSort(max_age=30, n_init=1)

    # Okno
    if args.display:
        cv2.namedWindow("Dron - Vivotek", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dron - Vivotek", 640, 480)

    times = deque(maxlen=30)
    frame_idx = 0

    print("▶️ Běží sledování... (Q/Esc ukončí)")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("❗ Stream ukončen nebo chyba čtení.")
            break

        frame_idx += 1
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        dets = extract_detections_from_result(results[0]) if results else []

        try:
            tracks = tracker.update_tracks(dets, frame=frame)
        except Exception as e:
            print("Chyba trackeru:", e)
            tracks = []

        det_count = 0
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            tlbr = tr.to_tlbr()
            x1, y1, x2, y2 = map(int, tlbr)
            w_box, h_box = x2 - x1, y2 - y1
            cls = getattr(tr, "det_class", None)
            if cls is None and hasattr(tr, "get_det_class"):
                try:
                    cls = tr.get_det_class()
                except:
                    cls = "?"
            conf = getattr(tr, "det_conf", 0.0) or 0.0
            label = f"ID:{tid} {cls} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            det_count += 1

        # FPS a info
        t1 = time.time()
        times.append(t1 - t0)
        fps_sm = len(times) / sum(times)
        cv2.putText(frame, f"FPS: {fps_sm:.1f}  Obj:{det_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.display:
            cv2.imshow("Dron - Vivotek", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (81, ord('a')):
            ptz.left()
        elif key in (83, ord('d')):
            ptz.right()
        elif key in (82, ord('w')):
            ptz.up()
        elif key in (84, ord('s')):
            ptz.down()
        elif key in (ord('p'), ord('P')):
            ptz.stop()

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Hotovo — zpracováno:", frame_idx, "snímků")


if __name__ == "__main__":
    main()
