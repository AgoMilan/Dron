# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:48:17 2025

@author: Milan

YOLO11 + FaceRecognitionManager v2.9.1 (2025-10)
------------------------------------------------
- Detekce obličejů pomocí YOLO11
- Sledování osob podle centroidů
- Integrace s FaceRecognitionManager
- Nově: možnost ručně zadat jméno známé osoby (klávesa 'N')
"""

import cv2
import torch
import argparse
import time
from ultralytics import YOLO
from visitor_db_v2 import VisitorDB
from face_recognition_manager_v2_9fin import FaceRecognitionManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--model", type=str, default="YOLO11face.pt")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--revisit_timeout", type=int, default=900)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MAIN] Device: {device}")

    model = YOLO(args.model)
    print(f"[MAIN] Načítám model: {args.model}")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[MAIN] Nelze otevřít zdroj videa.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[MAIN] Kamera {width}x{height} @ {fps:.1f}FPS")

    vdb = VisitorDB("visitors.json")
    frm = FaceRecognitionManager(device=device, visitor_db=vdb, decision_min_samples=2, auto_add_threshold=2)

    print("[MAIN] Klávesy: Q/Esc=ukončit, N=pojmenovat osobu")

    # --- jednoduché sledování podle centroidů ---
    tracks = {}
    track_id_counter = 0
    MAX_TRACK_DIST = 80
    TRACK_MAX_AGE = 1.5

    last_face_crop = None  # uloží poslední detekovaný obličej pro případné ruční přidání

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False) # verbose=False vypne výpis pro každý snímek
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) > 0 else []

        now_ts = time.time()
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detections.append({"bbox": (x1, y1, x2, y2), "center": (cx, cy)})

        for det in detections:
            cx, cy = det["center"]
            best_tid, best_dist = None, 1e9
            for tid, tinfo in tracks.items():
                tx, ty = tinfo["centroid"]
                dist = (tx - cx) ** 2 + (ty - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            if best_tid is None or (best_dist ** 0.5) > MAX_TRACK_DIST:
                track_id_counter += 1
                tid = track_id_counter
            else:
                tid = best_tid
            tracks[tid] = {"centroid": (cx, cy), "last_seen": now_ts}

            x1, y1, x2, y2 = det["bbox"]
            label, score, visitor_info = frm.update_track(frame, tid, (x1, y1, x2, y2))

            # uloží poslední obličej pro případ ručního pojmenování
            if "unknown" in label:
                last_face_crop = frm.crop_face(frame, (x1, y1, x2, y2))

            color = (0, 255, 0) if "unknown" not in label else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if visitor_info:
                vid = visitor_info.get("id", "?")
                visits = visitor_info.get("visits", 1)
                cv2.putText(frame, f"{vid} visits:{visits}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # mazání starých tracků
        for tid, tinfo in list(tracks.items()):
            if now_ts - tinfo["last_seen"] > TRACK_MAX_AGE:
                tracks.pop(tid, None)
                frm.track_buffers.pop(tid, None)
                frm.track_last_seen.pop(tid, None)
                frm.track_unknown_counts.pop(tid, None)

        if args.display:
            cv2.imshow("YOLO Face Tracker", frame)
            key = cv2.waitKey(1) & 0xFF

            # ukončení programu
            if key in [27, ord("q"), ord("Q")]:
                break

            # ruční přidání známé osoby
            elif key in [ord("n"), ord("N")]:
                if last_face_crop is not None:
                    name = input("Zadej jméno nové osoby: ").strip()
                    if name:
                        emb = frm.get_embedding(last_face_crop)
                        frm.enroll_person(name, emb, face_crop=last_face_crop)
                        print(f"[MAIN] Osoba '{name}' byla přidána do galerie.")
                        last_face_crop = None
                else:
                    print("[MAIN] Žádný aktuální obličej k přidání.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
