# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 12:22:53 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
YOLOv8 + DeepSORT sledování s ručním výběrem cíle, parametry převzaté z track_drone23.py
Autor: Milan + GPT-5
"""

"""
YOLO + DeepSORT tracker s pauzou pro výběr objektu a možností zmenšení bounding boxu (--shrink).
"""

import cv2
import torch
import os
import csv
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def main(args):
    print(f"Načítám model: {args.model}")
    model = YOLO(args.model)

    print("Inicializuji DeepSort...")
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.iou_dist,
    )

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Zdroj: {args.source}  {height}x{width} @ {fps:.2f}fps")

    # Výstupní video
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    csv_path = os.path.splitext(args.output)[0] + ".csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "confidence", "class"])

    selected_id = None
    frame_idx = 0
    shrink = getattr(args, "shrink", 0.0)
    start_time = time.time()

    scale = args.winsize / width
    print(f"Škála: {scale:.4f} -> zobrazím {args.winsize}x{int(height * scale)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # YOLO inference
        results = model.predict(frame, conf=getattr(args, "yolo_conf", getattr(args, "conf", 0.25)),
                        imgsz=getattr(args, "imgsz", 640), verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        vis = frame.copy()

        # --- Kreslení boxů ---
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            try:
                l, t, r, b = tr.to_ltrb()
                w, h = r - l, b - t
                cx, cy = l + w / 2, t + h / 2

                # Shrink kolem středu
                new_w = w * (1 - shrink)
                new_h = h * (1 - shrink)
                l = int(cx - new_w / 2)
                r = int(cx + new_w / 2)
                t = int(cy - new_h / 2)
                b = int(cy + new_h / 2)

                color = (0, 255, 0) if tr.track_id != selected_id else (0, 0, 255)
                cv2.rectangle(vis, (l, t), (r, b), color, 2)
                cv2.putText(vis, f"ID:{tr.track_id}", (l, max(12, t - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                csv_writer.writerow([frame_idx, tr.track_id, l, t, r, b,
                                     getattr(tr, "confidence", 0.0), getattr(tr, "cls", -1)])
            except Exception as e:
                print(f"⚠️ Chyba při kreslení boxu: {e}")
                continue

        # --- Pauza pro výběr objektu ---
        if frame_idx == args.pause_frame:
            print(f"⏸ Pauza na framu {frame_idx} – klikni na objekt nebo stiskni [A]/[Enter]/[Space]")

            paused = True
            click_pos = []

            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    click_pos.append((int(x / scale), int(y / scale)))
                    print(f"🖱 Klik (display): ({x},{y}) → (orig): ({int(x / scale)},{int(y / scale)})")

            cv2.namedWindow("YOLO DeepSORT")
            cv2.setMouseCallback("YOLO DeepSORT", click_event)

            while paused:
                disp = cv2.resize(vis, (args.winsize, int(height * scale)))
                cv2.imshow("YOLO DeepSORT", disp)
                key = cv2.waitKey(50) & 0xFF

                if key in [ord('q'), 27]:
                    print("🛑 Ukončeno uživatelem (q/esc).")
                    cap.release()
                    out.release()
                    csv_file.close()
                    cv2.destroyAllWindows()
                    return
                elif key in [ord('a'), ord(' '), 13]:
                    if len(tracks) > 0:
                        selected_id = tracks[0].track_id
                        print(f"✅ Auto-vybrán track ID {selected_id}")
                    paused = False
                elif click_pos:
                    x, y = click_pos[-1]
                    for tr in tracks:
                        l, t, r, b = tr.to_ltrb()
                        if l <= x <= r and t <= y <= b:
                            selected_id = tr.track_id
                            print(f"✅ Vybrán klikem ID {selected_id}")
                            paused = False
                            break
                    click_pos.clear()

        # --- Zobrazení ---
        if args.show:
            disp = cv2.resize(vis, (args.winsize, int(height * scale)))
            cv2.imshow("YOLO DeepSORT", disp)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        out.write(vis)

    # --- Ukončení ---
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"✅ Hotovo — frames: {frame_idx}, čas: {elapsed:.1f}s, FPS: {frame_idx / elapsed:.2f}")
    print(f"📹 Video: {args.output}")
    print(f"📄 CSV: {csv_path}")


if __name__ == "__main__":
    print("❌ Tento soubor není určen pro přímé spuštění. Použij main.py.")
