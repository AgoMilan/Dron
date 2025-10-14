# -*- coding: utf-8 -*-
"""
YOLO11face + DeepSORT pro sledování obličejů (bez PTZ)
-------------------------------------------------
✅ Sleduje pouze třídu 'face'
✅ Boxy zmenšené (~30 % původní velikosti)
✅ Boxy jsou čtvercové a vycentrované na obličej
✅ Zobrazuje FPS a počet sledovaných obličejů
"""

import argparse
import time
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def try_open_rtsp(rtsp_url, test_frames=3):
    """Ověří dostupnost RTSP streamu."""
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
    """
    Vrací pouze detekce obličejů.
    Boxy jsou zmenšené (~20 %) a posunuté nahoru na oblast obličeje.
    """
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

        # ✅ Jen obličeje
        if "face" not in cls_name.lower():
            continue

        # --- výpočet zmenšeného a posunutého boxu ---
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + int(h * 0.25)  # posun výš k hlavě

        shrink = 0.2  # velikost výsledného boxu (20 % původní oblasti)
        side = int(min(w, h) * shrink)

        # souřadnice čtverce
        x1n = max(0, cx - side // 2)
        y1n = max(0, cy - side // 2)
        x2n = min(int(res.orig_shape[1]), cx + side // 2)
        y2n = min(int(res.orig_shape[0]), cy + side // 2)

        dets.append(([x1n, y1n, x2n, y2n], conf, "face"))

    return dets



def main():
    parser = argparse.ArgumentParser(description="YOLO11face + DeepSORT tracker pro obličeje (čtvercové boxy)")
    parser.add_argument("--source", type=str, required=True, help="RTSP URL nebo index kamery (0)")
    parser.add_argument("--model", type=str, default="YOLO11face.pt", help="YOLO11 model pro detekci obličejů")
    parser.add_argument("--conf", type=float, default=0.4, help="Práh detekce (0-1)")
    parser.add_argument("--imgsz", type=int, default=640, help="Velikost obrazu pro YOLO")
    parser.add_argument("--device", type=str, default="cpu", help="Zařízení (cpu/cuda)")
    parser.add_argument("--display", action="store_true", help="Zobrazit výsledek")
    args = parser.parse_args()

    print("Načítám model:", args.model)
    model = YOLO(args.model)

    # Kamera / RTSP
    cap = cv2.VideoCapture(int(args.source)) if args.source.isdigit() else try_open_rtsp(args.source)
    if not cap or not cap.isOpened():
        raise RuntimeError("❌ Nelze otevřít zdroj videa.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Zdroj: {w}x{h} @ {fps:.1f} fps")

    # DeepSORT tracker
    tracker = DeepSort(max_age=30, n_init=1)

    if args.display:
        cv2.namedWindow("YOLO11 Face Tracker v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO11 Face Tracker v2", 640, 480)

    times = deque(maxlen=30)
    frame_idx = 0
    print("▶️ Běží sledování obličejů (čtvercové boxy)... (Q/Esc ukončí)")

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
            x1, y1, x2, y2 = map(int, tr.to_tlbr())
            conf = getattr(tr, "det_conf", 0.0) or 0.0
            label = f"ID:{tid} face {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            det_count += 1

        # FPS
        t1 = time.time()
        times.append(t1 - t0)
        fps_sm = len(times) / sum(times)
        cv2.putText(frame, f"FPS: {fps_sm:.1f}  Faces:{det_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.display:
            cv2.imshow("YOLO11 Face Tracker v2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Hotovo — zpracováno:", frame_idx, "snímků")


if __name__ == "__main__":
    main()
