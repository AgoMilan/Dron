# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 22:12:14 2025

@author: Milan
"""

# yolo11_face_tracker_v2_7.py
# -*- coding: utf-8 -*-
"""
Tracker v2.7 - integrace visitor DB + face_recognition_manager_v4
- uložit do Prodejna_2 a spustit tam
"""

import os
import argparse
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from visitor_db import VisitorDB
from face_recognition_manager_v2_7 import FaceRecognitionManager


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
        # scale
        x1 *= frame_w / img_w
        x2 *= frame_w / img_w
        y1 *= frame_h / img_h
        y2 *= frame_h / img_h
        dets.append(([x1, y1, x2, y2], conf, "face"))
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="rtsp://root:heslo@192.168.0.205/live.sdp")
    parser.add_argument("--model", type=str, default="YOLO11face.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--shrink", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gallery", type=str, default="gallery.json")
    parser.add_argument("--visitors", type=str, default="visitors.json")
    parser.add_argument("--auto_add_threshold", type=int, default=2, help="po kolika výskytech neznámého se založí visitor")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    print(f"[MAIN] device: {device}")
    print(f"[MAIN] loading model {args.model}")
    model = YOLO(args.model)
    model.to(device)

    # open camera / RTSP
    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        print("[MAIN] Nelze otevřít zdroj.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[MAIN] Camera {frame_w}x{frame_h} @ {fps_cam:.1f}FPS")

    # winsize for display
    winsize = int(frame_w * 0.7)

    # init modules
    visitor_db = VisitorDB(path=args.visitors, threshold=0.6)
    frm = FaceRecognitionManager(
        device=device,
        gallery_path=args.gallery,
        decision_min_samples=3,
        threshold=0.65,
        crop_margin=0.25,
        visitor_db=visitor_db,
        auto_add_threshold=args.auto_add_threshold
    )

    tracker = DeepSort(max_age=3, n_init=1, max_iou_distance=0.6)

    if args.display:
        cv2.namedWindow("Tracker v2.7", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracker v2.7", winsize, int(winsize * frame_h / frame_w))

    last_unknown_emb = None
    last_unknown_vid = None
    last_rec_info = None  # info dict from manager

    print("[MAIN] Spuštěno. Klávesy: E=enroll, D=delete, G=gallery, V=visitors, R=rename, Q/Esc=quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            t0 = time.time()
            results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
            res0 = results[0] if results else None
            dets = extract_face_detections(res0, frame_w, frame_h) if res0 is not None else []

            tracks = tracker.update_tracks(dets, frame=frame)
            display = frame.copy()
            scale = winsize / frame_w

            last_unknown_emb = None
            last_unknown_vid = None
            last_rec_info = None

            # draw YOLO dets (blue)
            for (x1, y1, x2, y2), conf, cls in dets:
                p1 = (int(x1 * scale), int(y1 * scale))
                p2 = (int(x2 * scale), int(y2 * scale))
                cv2.rectangle(display, p1, p2, (255, 0, 0), 1)

            for tr in tracks:
                if not tr.is_confirmed() or tr.time_since_update > 1:
                    continue
                # tr.to_tlbr returns floats
                x1, y1, x2, y2 = map(float, tr.to_tlbr())
                # compute square around center, apply shrink only to visual box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                side = min((x2 - x1), (y2 - y1))
                # full box for recognition (use full detection)
                full_box = (x1, y1, x2, y2)
                # visual box (square shrunk)
                vs = side * args.shrink
                vx1, vy1, vx2, vy2 = cx - vs / 2.0, cy - vs / 2.0, cx + vs / 2.0, cy + vs / 2.0
                # clip
                vx1, vy1 = max(0.0, vx1), max(0.0, vy1)
                vx2, vy2 = min(frame_w - 1.0, vx2), min(frame_h - 1.0, vy2)

                # ask manager to update and possibly interact with visitor DB
                label_text, score, visitor_info = frm.update_track(frame, tr.track_id, full_box)

                # draw visual square
                p1 = (int(vx1 * scale), int(vy1 * scale))
                p2 = (int(vx2 * scale), int(vy2 * scale))
                color = (0, 200, 0)
                # color coding: known=bright green, repeat visitor=cyan, unknown=orange
                if label_text.startswith("unknown"):
                    color = (0, 140, 255)
                elif visitor_info is not None:
                    color = (200, 200, 0)  # yellowish for visitor repeat
                else:
                    color = (0, 255, 0)  # known

                cv2.rectangle(display, p1, p2, color, 2)
                # label text
                cv2.putText(display, f"ID:{tr.track_id} {label_text}", (p1[0], max(20, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # store last unknown/visitor info for manual actions
                if label_text.startswith("unknown"):
                    # keep last avg emb if any
                    buf = frm.track_buffers.get(tr.track_id, None)
                    if buf is not None and len(buf) > 0:
                        last_unknown_emb = np.mean(np.stack(list(buf)), axis=0)
                if visitor_info is not None:
                    last_unknown_vid = visitor_info.get("id")
                    last_rec_info = visitor_info

            # draw fps
            fps = 1.0 / max((time.time() - t0), 1e-6)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if args.display:
                cv2.imshow("Tracker v2.7", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break

            elif key in (ord('e'), ord('E')):
                # enroll: if last_unknown_emb present, ask name; allow Enter to repeat last name
                if last_unknown_emb is None:
                    print("[ENROLL] Žádný vhodný obličej pro enroll.")
                else:
                    print("\n[ENROLL] Zadej jméno osoby (Enter = zrušit): ", end="", flush=True)
                    try:
                        name = input().strip()
                    except Exception:
                        name = ""
                    if name:
                        frm.enroll_person(name, last_unknown_emb)
                        # optionally, if the person existed as unknown visitor, we can delete visitor entry or rename
                        print(f"[ENROLL] Přidáno do gallery: {name}")
                    else:
                        print("[ENROLL] Zrušeno.")

            elif key in (ord('d'), ord('D')):
                # delete: try gallery first, then visitors
                print("\n[DELETE] Zadej jméno (pro gallery) nebo visitor id (unknown_XXX) k odstranění (Enter = zrušit): ", end="", flush=True)
                try:
                    tok = input().strip()
                except Exception:
                    tok = ""
                if not tok:
                    print("[DELETE] Zrušeno.")
                else:
                    ok = frm.delete_person(tok)
                    if ok:
                        print(f"[DELETE] {tok} smazáno z gallery.")
                    else:
                        ok2 = visitor_db.delete(tok)
                        if ok2:
                            print(f"[DELETE] {tok} smazáno z visitors.")
                        else:
                            print(f"[DELETE] Nalezeno nikde: {tok}")

            elif key in (ord('g'), ord('G')):
                names = frm.list_gallery()
                print("\n[GALLERY] known persons:")
                if names:
                    for n in names:
                        print("  -", n)
                else:
                    print("  (prázdná)")

            elif key in (ord('v'), ord('V')):
                # list visitors
                vis = visitor_db.list_visitors()
                print("\n[VISITORS] id, label, visits:")
                if vis:
                    for vid, label, visits in vis:
                        print(f"  {vid}  {label}  visits:{visits}")
                else:
                    print("  (žádní)")

            elif key in (ord('r'), ord('R')):
                # rename visitor -> prompt id and new name, optionally move to gallery
                print("\n[RENAME] Zadej visitor id (např. unknown_001): ", end="", flush=True)
                try:
                    vid = input().strip()
                except Exception:
                    vid = ""
                if not vid:
                    print("[RENAME] Zrušeno.")
                else:
                    rec = visitor_db.get_record(vid)
                    if rec is None:
                        print(f"[RENAME] {vid} nenalezen.")
                    else:
                        print(f"[RENAME] Zadej nové jméno pro {vid} (Enter = zrušit): ", end="", flush=True)
                        try:
                            newname = input().strip()
                        except Exception:
                            newname = ""
                        if not newname:
                            print("[RENAME] Zrušeno.")
                        else:
                            # option: move to gallery (copy embeddings)
                            print("[RENAME] Chceš zkopírovat embeddings do gallery (y/n)? ", end="", flush=True)
                            try:
                                yn = input().strip().lower()
                            except Exception:
                                yn = "n"
                            # rename visitor
                            visitor_db.rename(vid, newname)
                            print(f"[RENAME] {vid} přejmenován na {newname}")
                            if yn.startswith('y'):
                                rec2 = visitor_db.get_record(vid)
                                if rec2:
                                    # enroll all embeddings into gallery under newname
                                    for e in rec2["embeddings"]:
                                        frm.enroll_person(newname, e)
                                    print(f"[RENAME] embeddings z {vid} zkopírovány do gallery jako '{newname}'")

            # housekeeping
            frm.cleanup()

    except KeyboardInterrupt:
        print("[MAIN] Ukončeno uživatelem (CTRL+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[MAIN] Konec.")


if __name__ == "__main__":
    main()
