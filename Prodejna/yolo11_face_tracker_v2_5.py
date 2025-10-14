# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:07:50 2025

@author: Milan
"""

# yolo11_face_tracker_v2_5.py
# -*- coding: utf-8 -*-
"""
YOLO11face + DeepSORT + FaceRecognitionManager (v2_5)
- Přidána správa galerie z UI:
    E = enroll poslední unknown
    D = delete (smaže jméno z gallery.json)
    G = vypíše seznam jmen v galerii
- Zachováno: čtvercové boxy, shrink, RTSP, okamžité mizení tracků
"""

import argparse
import time
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from face_recognition_manager_v1 import FaceRecognitionManager


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
        cls_name = res.names.get(int(clss[i]), str(int(clss[i]))) if hasattr(res, "names") else "face"
        if "face" not in cls_name.lower():
            continue
        x1 *= frame_w / img_w
        x2 *= frame_w / img_w
        y1 *= frame_h / img_h
        y2 *= frame_h / img_h
        dets.append(([x1, y1, x2, y2], conf, "face"))
    return dets


def main():
    parser = argparse.ArgumentParser(description="YOLO11 Face Tracker v2_5 (RTSP + Recognition + Gallery manage)")
    parser.add_argument("--source", type=str, default="rtsp://root:heslo@192.168.0.205/live.sdp",
                        help="RTSP stream kamery nebo index (0)")
    parser.add_argument("--model", type=str, default="YOLO11face.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")
    parser.add_argument("--display", action="store_true", help="zobrazit okno")
    parser.add_argument("--winsize", type=str, default="auto", help="velikost okna (pixels)")
    parser.add_argument("--shrink", type=float, default=0.55, help="zmenšení vizuálního čtverce (0–1)")
    parser.add_argument("--gallery", type=str, default="gallery.json", help="cesta ke gallery.json")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Zařízení: {device}")

    print(f"[INFO] Načítám YOLO model: {args.model}")
    model = YOLO(args.model).to(device)

    # otevření kamery
    source = args.source
    cap = cv2.VideoCapture(0 if source.isdigit() else source)
    if not cap.isOpened():
        print("[ERROR] Nelze otevřít zdroj videa.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[INFO] Kamera: {frame_w}x{frame_h} @ {fps_cam:.1f} FPS")

    winsize = int(frame_w * 0.7) if args.winsize == "auto" else int(args.winsize)
    print(f"[INFO] winsize = {winsize}")

    tracker = DeepSort(max_age=3, n_init=1, embedder="mobilenet", max_iou_distance=0.6)
    frm = FaceRecognitionManager(device=device, gallery_path=args.gallery, crop_margin=0.25)

    unknown_dir = "unknowns"
    os.makedirs(unknown_dir, exist_ok=True)

    win_name = "YOLO11 Face Tracker v2_5"
    if args.display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, winsize, int(winsize * frame_h / frame_w))

    print("[INFO] Spuštěno.")
    print("  Klávesy: E = enroll, D = delete jméno z galerie, G = vypiš galerii, Q/Esc = ukončit")

    last_unknown_crop = None
    last_unknown_emb = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            t0 = time.time()
            results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
            res0 = results[0] if results else None
            dets = extract_face_detections(res0, frame_w, frame_h)

            tracks = tracker.update_tracks(dets, frame=frame)
            display = frame.copy()
            scale = winsize / frame_w

            # vykreslit raw detekce (modré)
            for (x1, y1, x2, y2), conf, cls in dets:
                p1 = (int(x1 * scale), int(y1 * scale))
                p2 = (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(display, p1, p2, (255, 0, 0), 1)

            last_unknown_crop = None
            last_unknown_emb = None

            # zpracování tracků
            for tr in tracks:
                # přeskočit nepotvrzené a staré tracky
                if not tr.is_confirmed() or tr.time_since_update > 1:
                    continue

                tid = tr.track_id
                x1, y1, x2, y2 = map(float, tr.to_tlbr())

                # rozpoznávání na plné velikosti (neovlivní vizuální shrink)
                label_text, score = frm.update_track(frame, tid, (x1, y1, x2, y2))

                # vizuální čtverec (shrink řídí vizuální box)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                side = min((x2 - x1), (y2 - y1)) * args.shrink
                if side <= 0:
                    sx1, sy1, sx2, sy2 = x1, y1, x2, y2
                else:
                    half = side / 2.0
                    sx1, sy1, sx2, sy2 = cx - half, cy - half, cx + half, cy + half

                # clip
                sx1, sy1 = max(0.0, sx1), max(0.0, sy1)
                sx2, sy2 = min(frame_w - 1.0, sx2), min(frame_h - 1.0, sy2)

                p1 = (int(sx1 * scale), int(sy1 * scale))
                p2 = (int(sx2 * scale), int(sy2 * scale))
                cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

                cv2.putText(display, f"ID:{tid} {label_text}", (p1[0], max(20, p1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # pokud unknown, uložíme průměrný embedding a crop pro enroll
                if label_text.startswith("unknown"):
                    buf = frm.track_buffers.get(tid, None)
                    if buf is not None and len(buf) > 0:
                        try:
                            arr = np.stack(list(buf), axis=0)
                            avg_emb = arr.mean(axis=0)
                            avg_emb /= (np.linalg.norm(avg_emb) + 1e-6)
                            last_unknown_emb = avg_emb
                            last_unknown_crop = frm.crop_face(frame, (x1, y1, x2, y2))
                        except Exception:
                            last_unknown_emb = None
                            last_unknown_crop = None

            # FPS
            fps = 1.0 / max((time.time() - t0), 1e-5)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if args.display:
                cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

            elif key in (ord('e'), ord('E')):
                # enroll posledního unknown
                if last_unknown_emb is not None and last_unknown_crop is not None:
                    try:
                        preview = cv2.resize(last_unknown_crop, (200, 200))
                        cv2.imshow("Enroll preview", preview)
                    except Exception:
                        pass
                    print("\n[ENROLL] Zadej jméno (Enter = zrušit): ", end="", flush=True)
                    try:
                        name = input().strip()
                    except Exception:
                        name = ""
                    if name:
                        frm.enroll_person(name, last_unknown_emb)
                        fname = os.path.join(unknown_dir, f"{name}_{int(time.time())}.jpg")
                        try:
                            cv2.imwrite(fname, last_unknown_crop)
                            print(f"[ENROLL] Uloženo -> {fname}")
                        except Exception as e:
                            print(f"[ENROLL] Chyba při ukládání náhledu: {e}")
                    else:
                        print("[ENROLL] Zrušeno.")
                    try:
                        cv2.destroyWindow("Enroll preview")
                    except Exception:
                        pass
                else:
                    print("[ENROLL] Žádný vhodný unknown pro enroll.")

            elif key in (ord('d'), ord('D')):
                # delete jména z galerie
                names = frm.list_gallery()
                if len(names) == 0:
                    print("[DELETE] Galerie je prázdná.")
                else:
                    print("\n[DELETE] Galerie (seznam jmen):")
                    for n in names:
                        print("  -", n)
                    print("[DELETE] Zadej jméno k smazání (Enter = zrušit): ", end="", flush=True)
                    try:
                        name = input().strip()
                    except Exception:
                        name = ""
                    if name:
                        ok = frm.delete_person(name)
                        if ok:
                            print(f"[DELETE] '{name}' smazáno.")
                        else:
                            print(f"[DELETE] '{name}' nenalezeno.")
                    else:
                        print("[DELETE] Zrušeno.")

            elif key in (ord('g'), ord('G')):
                # vypiš galerii
                names = frm.list_gallery()
                if len(names) == 0:
                    print("[GALLERY] Galerie je prázdná.")
                else:
                    print("\n[GALLERY] Seznam jmen v galerii:")
                    for n in names:
                        print("  -", n)

            # housekeeping
            frm.cleanup()

    except KeyboardInterrupt:
        print("[INFO] Ukončeno uživatelem (CTRL+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Program ukončen.")


if __name__ == "__main__":
    main()
