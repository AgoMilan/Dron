# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 22:43:32 2025

@author: Milan

YOLO11face + DeepSORT v10 – automaticky kalibrované sledování obličejů
----------------------------------------------------------------------
✅ Automatický výběr optimálního měřítka zobrazení (--winsize auto)
✅ Přesný přepočet souřadnic (žádný posun rámečků)
✅ Čtvercové boxy zarovnané na obličej
✅ Modrý = YOLO detekce, Zelený = DeepSORT sledování
"""

import argparse
import time
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ===============================
# Detekce obličejů s přepočtem na původní rozlišení
# ===============================
def extract_face_detections(res, frame_w, frame_h, shrink=0.5):
    """Vrací čtvercové boxy přepočítané na původní rozlišení."""
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    img_w, img_h = res.orig_shape[1], res.orig_shape[0]
    scale_x = frame_w / img_w
    scale_y = frame_h / img_h

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4]
        conf = float(confs[i])
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)

        if "face" not in cls_name.lower():
            continue

        # Přepočet na původní rozlišení videa
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = min((x2 - x1), (y2 - y1)) * shrink
        x1n, y1n = cx - side / 2, cy - side / 2
        x2n, y2n = cx + side / 2, cy + side / 2

        dets.append(([x1n, y1n, x2n, y2n], conf, "face"))
    return dets


# ===============================
# Hlavní funkce
# ===============================
def main():
    parser = argparse.ArgumentParser(description="YOLO11face + DeepSORT – přesné sledování obličejů")
    parser.add_argument("--source", type=str, required=True, help="RTSP URL nebo index kamery (0)")
    parser.add_argument("--model", type=str, default="YOLO11face.pt")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--winsize", type=str, default="auto", help="Šířka okna (nebo 'auto' pro automatické měřítko)")
    parser.add_argument("--shrink", type=float, default=0.4)
    args = parser.parse_args()

    # Výběr zařízení
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Načítám model: {args.model} ({device.upper()})")

    model = YOLO(args.model)
    model.to(device)

    cap = cv2.VideoCapture(int(args.source)) if args.source.isdigit() else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("❌ Nelze otevřít video zdroj.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Zdroj otevřen: {w}x{h} @ {fps_cam:.1f} FPS")

    # ===============================
    # Automatický výpočet winsize (kalibrace)
    # ===============================
    if args.winsize == "auto":
        # ideální měřítko ~0.68 reálného rozlišení
        winsize = int(w * 0.68)
        print(f"🧩 Automaticky nastaveno winsize = {winsize}px (pro {w}x{h})")
    else:
        winsize = int(args.winsize)
        print(f"Používám ručně zadané winsize = {winsize}px")

    # Inicializace trackeru
    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet")

    # Okno
    if args.display:
        cv2.namedWindow("YOLO11 Face Tracker v10", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLO11 Face Tracker v10", winsize, int(winsize * h / w))

    print("▶️ Sledování obličejů spuštěno (Q/Esc ukončí)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❗ Konec streamu nebo chyba čtení.")
            break

        start_t = time.time()

        # YOLO detekce
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
        dets = extract_face_detections(results[0], w, h, args.shrink) if results else []

        # Aktualizace trackeru
        tracks = tracker.update_tracks(dets, frame=frame)

        display = frame.copy()
        scale = winsize / w

        # --- Modré boxy (YOLO detekce) ---
        for (x1, y1, x2, y2), conf, cls in dets:
            p1 = (int(x1 * scale), int(y1 * scale))
            p2 = (int(x2 * scale), int(y2 * scale))
            cv2.rectangle(display, p1, p2, (255, 0, 0), 2)
            cv2.putText(display, f"{cls} {conf:.2f}", (p1[0], max(20, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- Zelené boxy (DeepSORT) ---
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = map(float, tr.to_tlbr())
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side = min((x2 - x1), (y2 - y1))
            x1n, y1n = cx - side / 2, cy - side / 2
            x2n, y2n = cx + side / 2, cy + side / 2

            p1 = (int(x1n * scale), int(y1n * scale))
            p2 = (int(x2n * scale), int(y2n * scale))
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)
            cv2.putText(display, f"ID:{tr.track_id}", (p1[0], max(20, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS
        fps = 1.0 / max(1e-5, (time.time() - start_t))
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Zobrazení
        if args.display:
            cv2.imshow("YOLO11 Face Tracker v10", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Ukončeno.")


if __name__ == "__main__":
    main()
