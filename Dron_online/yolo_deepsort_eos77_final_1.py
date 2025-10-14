# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 01:26:25 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
yolo_deepsort_eos77_final_v7_airplane_fixed_box.py
--------------------------------------------------
YOLOv8 + DeepSORT
Detekce pouze 'airplane', správné souřadnice (bez posunu), název objektu, confidence a FPS overlay.
"""

import argparse
import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ---------------------------------------------------------------------------
# Přesná konverze souřadnic (odstraní posun boxu)
# ---------------------------------------------------------------------------
def extract_detections_from_result(res, frame_w, frame_h, target_class="airplane"):
    """Správně přepočítá YOLOv8 detekce na původní rozměr snímku a filtruje požadovanou třídu."""
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    # Rozměry vstupního snímku YOLO
    img_h, img_w = res.orig_shape

    # Pokud YOLO interně měnilo poměr stran (letterbox), přepočítáme zpět
    gain = min(frame_w / img_w, frame_h / img_h)
    pad_x = (frame_w - img_w * gain) / 2
    pad_y = (frame_h - img_h * gain) / 2

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4]
        conf = float(confs[i])
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id))

        # Filtr: jen daná třída
        if cls_name.lower() != target_class.lower():
            continue

        # Převod z YOLO souřadnic zpět do originálu (kompenzace paddingu)
        x1 = (x1 - pad_x) / gain
        y1 = (y1 - pad_y) / gain
        x2 = (x2 - pad_x) / gain
        y2 = (y2 - pad_y) / gain

        # Ořez na hranice
        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        if x2 > x1 and y2 > y1:
            dets.append(([x1, y1, x2, y2], conf, cls_name))

    return dets


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Index kamery nebo cesta k videu")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--winsize", type=int, default=1280)
    parser.add_argument("--shrink", type=float, default=0.5)
    parser.add_argument("--cls", type=str, default="airplane", help="Třída objektu k detekci")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    model = YOLO(args.model)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Nelze otevřít kameru.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Kamera otevřena {w}x{h}@{fps:.1f}")

    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet")

    cv2.namedWindow("EOS77D Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("EOS77D Live", args.winsize, int(args.winsize * h / w))

    prev_time = time.time()
    avg_fps = fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOLOv8 predikce ---
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False)
        dets = extract_detections_from_result(results[0], w, h, target_class=args.cls) if results else []

        # --- Filtrování ---
        valid_dets = []
        for (box, conf, cls_name) in dets:
            x1, y1, x2, y2 = map(int, box)
            if x2 > x1 + 2 and y2 > y1 + 2:
                valid_dets.append(([x1, y1, x2, y2], conf, cls_name))

        # --- DeepSORT tracking ---
        tracks = []
        if valid_dets:
            try:
                tracks = tracker.update_tracks(valid_dets, frame=frame)
            except Exception as e:
                print(f"⚠️ Chyba při update_tracks: {e}")
                tracks = []

        vis = frame.copy()

        # --- Kreslení boxů ---
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            try:
                l, t, r, b = map(int, tr.to_ltrb())
                w_box, h_box = r - l, b - t
                cx, cy = l + w_box / 2, t + h_box / 2

                shrink = args.shrink
                new_w = w_box * (1 - shrink)
                new_h = h_box * (1 - shrink)
                l = int(cx - new_w / 2)
                r = int(cx + new_w / 2)
                t = int(cy - new_h / 2)
                b = int(cy + new_h / 2)

                color = (0, 255, 0)
                conf = next((d[1] for d in valid_dets if abs(d[0][0] - l) < 20), 0)
                cls_name = next((d[2] for d in valid_dets if abs(d[0][0] - l) < 20), args.cls)

                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(vis, (l, t), (r, b), color, 2)
                cv2.putText(vis, label, (l, max(12, t - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                print(f"⚠️ Chyba při kreslení boxu: {e}")
                continue

        # --- FPS ---
        now = time.time()
        fps_now = 1 / (now - prev_time)
        prev_time = now
        avg_fps = 0.9 * avg_fps + 0.1 * fps_now
        cv2.putText(vis, f"FPS: {avg_fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # --- Zobrazení ---
        cv2.imshow("EOS77D Live", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
