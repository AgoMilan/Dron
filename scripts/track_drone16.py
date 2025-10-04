# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 22:26:33 2025

@author: Milan
"""

"""
track_drone16.py

Verze s vylepšenou predikcí trajektorie pro vybraný track pomocí Kalmanova filtru.
- Kalman sleduje (cx, cy, vx, vy, w, h) - stav: střed + rychlost + velikost.
- Měření jsou (cx, cy, w, h) z detekce/trackeru.
- Pokud tracker/YOLO momentálně nevidí vybraný objekt, použije se predict() z Kalmanu.
- Predikci omezíme maximální povolenou rychlostí (pixels/frame) aby se zabránilo "přeskoku".
"""

import cv2
import time
import csv
import argparse
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort

# Globální proměnné pro klikání/debug
selected_id = None
click_pos = None
click_debug = None


def mouse_callback(event, x, y, flags, param):
    """Callback pro kliknutí myší s přepočtem na originální souřadnice"""
    global click_pos, click_debug
    if event == cv2.EVENT_LBUTTONDOWN:
        if param is not None:
            scale_x = param.get("scale_x", 1.0)
            scale_y = param.get("scale_y", 1.0)
            x = int(x / scale_x)
            y = int(y / scale_y)
        click_pos = (x, y)
        click_debug = (x, y)
        print(f"🖱 Klik na (orig): {click_pos}")


def extract_dets_from_result(res):
    """Převede YOLO výstup na seznam (bbox, conf, cls)
    bbox jako [x1, y1, w, h] (x1,y1 = levý horní roh)
    """
    dets = []
    if res is None or res.boxes is None:
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()
    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, b[:4])
        w = x2 - x1
        h = y2 - y1
        dets.append(([x1, y1, w, h], float(confs[i]), int(clss[i])))
    return dets


class KalmanBBox:
    """
    Jednoduchý Kalmanův filtr pro sledování bounding boxu (vybraného objektu).

    Stav: [cx, cy, vx, vy, w, h]^T
    Měření: [cx, cy, w, h]^T

    Parametry:
      - process_noise: Q (malé hodnoty = méně důvěry v model predikce)
      - measurement_noise: R (malé hodnoty = více důvěry v měření)
      - max_vel: maximální povolená rychlost (pix/frame) pro ořez predikce
    """

    def __init__(self, process_noise=1e-2, measurement_noise=1e-1, max_vel=50.0):
        # state: 6, measurement:4
        self.kf = cv2.KalmanFilter(6, 4, 0, cv2.CV_32F)
        # Transition matrix A
        # x' = x + vx
        # y' = y + vy
        # vx' = vx
        # vy' = vy
        # w' = w
        # h' = h
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H (maps state to measurement)
        # meas = [cx, cy, w, h] -> pick corresponding state components
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # cx
            [0, 1, 0, 0, 0, 0],  # cy
            [0, 0, 0, 0, 1, 0],  # w
            [0, 0, 0, 0, 0, 1]   # h
        ], dtype=np.float32)

        # Noise covariances
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        # Posteriori error estimate
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.initialized = False
        self.max_vel = float(max_vel)

    def initialize(self, bbox):
        """Inicializace stavu z bbox (x1,y1,w,h) - převedeme na střed"""
        x1, y1, w, h = bbox
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        state = np.array([cx, cy, 0., 0., float(w), float(h)], dtype=np.float32).reshape(6, 1)
        self.kf.statePost = state
        self.initialized = True

    def correct(self, bbox):
        """Korekce Kalmanu měřením bbox (x1,y1,w,h)"""
        if not self.initialized:
            self.initialize(bbox)
            return None
        x1, y1, w, h = bbox
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        meas = np.array([cx, cy, float(w), float(h)], dtype=np.float32).reshape(4, 1)
        corrected = self.kf.correct(meas)
        # Vrátíme korektovanou bbox (x1,y1,w,h)
        cx_c, cy_c, _, _, w_c, h_c = corrected.flatten()
        x1_c = cx_c - w_c / 2.0
        y1_c = cy_c - h_c / 2.0
        return [int(x1_c), int(y1_c), int(w_c), int(h_c)]

    def predict(self):
        """Predikce a ořez rychlosti (aby se nepředbíhalo)"""
        if not self.initialized:
            return None
        predicted = self.kf.predict()
        cx, cy, vx, vy, w, h = predicted.flatten()
        # Omezíme rychlost
        vx = np.clip(vx, -self.max_vel, self.max_vel)
        vy = np.clip(vy, -self.max_vel, self.max_vel)
        # Pokud jsme omezili, zapíšeme zpět do stavu (aby to ovlivnilo další predikce)
        self.kf.statePre[2, 0] = vx
        self.kf.statePre[3, 0] = vy
        x1_p = cx - w / 2.0
        y1_p = cy - h / 2.0
        return [int(x1_p), int(y1_p), int(w), int(h)]

    def get_state(self):
        """Vrátí aktuální stav (cx,cy,vx,vy,w,h)"""
        if not self.initialized:
            return None
        return self.kf.statePost.flatten().tolist()


def main(args):
    global selected_id, click_pos, click_debug

    video_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # YOLO model
    model = YOLO(args.model)
    device = "cpu" if args.device == "cpu" else args.device

    # DeepSORT tracker s laditelnými parametry (ponecháno jak máš)
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.iou_dist
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Nelze otevřít video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # CSV log
    csv_file = out_path.with_suffix(".csv")
    csv_f = open(csv_file, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["frame", "track_id", "cx", "cy", "w", "h", "reason"])

    frame_idx = 0
    t0 = time.time()
    print("▶️ Start processing...", video_path)

    # Scale faktory pro klikání
    if args.winsize > 0:
        scale_x = args.winsize / w
        scale_y = args.winsize / h
    else:
        scale_x, scale_y = 1.0, 1.0

    if args.show:
        cv2.namedWindow("tracking")
        cv2.setMouseCallback("tracking", mouse_callback,
                             {"scale_x": scale_x, "scale_y": scale_y})

    vis = None
    tracks = []
    paused_for_selection = False

    # Kalman pro vybraný track (inicializujeme až po výběru)
    kf = None
    missing_steps = 0  # kolik snímků chybí detekce pro selected track

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detekce
        results = model.predict(source=[frame], conf=args.conf, imgsz=args.imgsz,
                                device=device, verbose=False)
        res = results[0] if results else None
        dets = extract_dets_from_result(res)

        # DeepSORT tracking
        tracks = tracker.update_tracks(dets, frame=frame)

        vis = frame.copy()
        annotator = Annotator(vis)

        # vykreslení YOLO boxů (etikety)
        if res and res.boxes is not None:
            for box in res.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                annotator.box_label(b, f"{model.names[cls]} {conf:.2f}")

        # Pauza pro ruční výběr (nebo auto fallback)
        if selected_id is None and tracks and frame_idx >= args.pause_frame:
            if not paused_for_selection:
                paused_for_selection = True

                # vykreslíme aktuální tracky zeleně
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    l, t_, r, b_ = map(int, t.to_ltrb())
                    cv2.rectangle(vis, (l, t_), (r, b_), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID:{t.track_id}", (l, max(15, t_ - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"⏸ Pauza na framu {frame_idx} – klikni na dron nebo klávesu pro auto-výběr")
                cv2.imshow("tracking", cv2.resize(vis, (args.winsize, args.winsize)) if args.winsize > 0 else vis)
                key = cv2.waitKey(0)

                if click_pos is not None:
                    x_click, y_click = click_pos
                    found = False
                    for t in tracks:
                        if not t.is_confirmed():
                            continue
                        l, t_, r, b_ = map(int, t.to_ltrb())
                        print(f"   Box ID {t.track_id}: L={l},T={t_},R={r},B={b_}")
                        if l <= x_click <= r and t_ <= y_click <= b_:
                            selected_id = t.track_id
                            print(f"✅ Ručně vybrán track ID {selected_id}")
                            found = True
                            break
                    if not found:
                        print("❌ Klik mimo box – proveden automatický výběr")
                    click_pos = None

                # fallback – automatický výběr jako dosud
                if selected_id is None and dets:
                    best_det = max(dets, key=lambda d: d[1])
                    bx, by, bw, bh = best_det[0]
                    det_conf = best_det[1]
                    for t in tracks:
                        if not t.is_confirmed():
                            continue
                        l, t_, r, b_ = map(int, t.to_ltrb())
                        if (bx < r and bx + bw > l and by < b_ and by + bh > t_):
                            selected_id = t.track_id
                            print(f"✅ Automaticky vybrán track ID {selected_id} (conf={det_conf:.2f})")
                            break

                # pokud máme selected_id, inicializujeme kalman z aktuální bbox (pokud dostupný)
                if selected_id is not None:
                    # najdeme track s tím ID a inicializujeme kalman
                    for t in tracks:
                        if t.track_id == selected_id and t.is_confirmed():
                            l, t_, r, b_ = map(int, t.to_ltrb())
                            bbox = [l, t_, r - l, b_ - t_]
                            kf = KalmanBBox(process_noise=args.kf_q, measurement_noise=args.kf_r,
                                            max_vel=args.kf_max_vel)
                            kf.initialize(bbox)
                            missing_steps = 0
                            print("🔧 Kalman inicializován z aktuální detekce.")
                            break

        # Pokud už máme selected_id, pracujeme s ním a Kalmanem
        if selected_id is not None:
            # Hledáme track s tímto ID
            matched_track = None
            for t in tracks:
                if not t.is_confirmed():
                    continue
                if t.track_id == selected_id:
                    matched_track = t
                    break

            if matched_track is not None:
                # máme měření -> korigujeme Kalman a kreslíme korektní bbox
                l, t_, r, b_ = map(int, matched_track.to_ltrb())
                meas_bbox = [l, t_, r - l, b_ - t_]
                bbox_corr = kf.correct(meas_bbox) if kf is not None else meas_bbox
                missing_steps = 0
                reason = "measured"
            else:
                # track s tím ID není (occluded) -> použijeme predikci z Kalmanu
                missing_steps += 1
                bbox_pred = kf.predict() if kf is not None else None
                reason = "predicted"
                if bbox_pred is not None:
                    bbox_corr = bbox_pred
                else:
                    bbox_corr = None

            # Pokud máme nějaký korr bbox, vykreslíme ho a uložíme do CSV
            if bbox_corr is not None:
                lx, ty, ww, hh = bbox_corr
                cx = int(lx + ww / 2)
                cy = int(ty + hh / 2)
                # vykreslení červeného boxu (sledovaný objekt)
                cv2.rectangle(vis, (int(lx), int(ty)), (int(lx + ww), int(ty + hh)), (0, 0, 255), 2)
                cv2.putText(vis, f"ID:{selected_id}", (int(lx), max(15, int(ty) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                csv_writer.writerow([frame_idx, selected_id, cx, cy, int(ww), int(hh), reason])

            # Pokud predikce selže nebo missing_steps překročí limit, ukončíme sledování
            if missing_steps > args.kf_max_pred_steps:
                print(f"⚠️ Selected track {selected_id} ztracen po {missing_steps} predikcích. Ukončuji sledování.")
                selected_id = None
                kf = None
                missing_steps = 0

        # vykreslení debug kliknutí
        if click_debug is not None:
            cv2.drawMarker(vis, click_debug, (0, 0, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)

        # zobrazení
        if args.show and vis is not None:
            show_frame = vis
            if args.winsize > 0:
                show_frame = cv2.resize(vis, (args.winsize, args.winsize))
            cv2.imshow("tracking", show_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()
    csv_f.close()
    cv2.destroyAllWindows()
    dt = time.time() - t0
    print(f"✅ Hotovo — frames: {frame_idx}, time: {dt:.1f}s, avg FPS: {frame_idx/dt:.2f}")
    print("📹 Výstupní video:", out_path)
    print("📄 CSV s logy:", csv_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="cesta k videu (mp4)")
    parser.add_argument("--output", type=str, default="runs/track_out.mp4", help="výstupní video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model (např. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="image size pro detekci")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' nebo '0' pro GPU")
    parser.add_argument("--show", action="store_true", help="zobrazit okno s náhledem")
    parser.add_argument("--winsize", type=int, default=0, help="velikost okna (šířka=výška), 0 = původní velikost")
    parser.add_argument("--pause_frame", type=int, default=0, help="Frame číslo pro ruční výběr (0 = začátek)")

    # DeepSORT tuning
    parser.add_argument("--max_age", type=int, default=60, help="Počet snímků, kdy může být track nezachycen, než se ukončí")
    parser.add_argument("--n_init", type=int, default=3, help="Počet po sobě jdoucích detekcí k potvrzení nového tracku")
    parser.add_argument("--iou_dist", type=float, default=0.7, help="Maximální IoU vzdálenost pro přiřazení tracku k detekci")

    # Kalman tuning (nové)
    parser.add_argument("--kf_q", type=float, default=1e-2, help="Kalman process noise (Q) - větší = více predikce")
    parser.add_argument("--kf_r", type=float, default=1e-1, help="Kalman measurement noise (R) - větší = méně důvěry v měření")
    parser.add_argument("--kf_max_vel", type=float, default=80.0, help="Max povolená rychlost predikce (pixels/frame)")
    parser.add_argument("--kf_max_pred_steps", type=int, default=60, help="Maximální počet predikcí bez měření před ukončením sledování")

    args = parser.parse_args()
    main(args)
