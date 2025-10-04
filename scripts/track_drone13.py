# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 21:24:06 2025

@author: Milan
"""

import cv2
import time
import csv
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from deep_sort_realtime.deepsort_tracker import DeepSort

# Globální proměnné
selected_id = None
click_pos = None
click_debug = None  # pro vykreslení kliknutí


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
    """Převede YOLO výstup na seznam (bbox, conf, cls)"""
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


def main(args):
    global selected_id, click_pos, click_debug

    video_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # YOLO model
    model = YOLO(args.model)
    device = "cpu" if args.device == "cpu" else args.device

    # DeepSORT tracker
    tracker = DeepSort(max_age=30)

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
    csv_writer.writerow(["frame", "track_id", "cx", "cy", "w", "h", "conf"])

    frame_idx = 0
    t0 = time.time()
    print("▶️ Start processing...", video_path)

    # Scale faktory pro klikání
    scale_x = w / (args.winsize if args.winsize > 0 else w)
    scale_y = h / (args.winsize if args.winsize > 0 else h)

    if args.show:
        cv2.namedWindow("tracking")
        cv2.setMouseCallback("tracking", mouse_callback,
                             {"scale_x": scale_x, "scale_y": scale_y})

    vis = None
    tracks = []
    paused_for_selection = False

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

        # vykreslení YOLO boxů
        if res and res.boxes is not None:
            for box in res.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                annotator.box_label(b, f"{model.names[cls]} {conf:.2f}")

        # 👉 Pauza pro výběr
        if selected_id is None and tracks and frame_idx >= args.pause_frame:
            if not paused_for_selection:
                paused_for_selection = True

                # Při pauze vykreslíme všechny aktuální tracky (zeleně)
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    l, t_, r, b_ = map(int, t.to_ltrb())
                    cv2.rectangle(vis, (l, t_), (r, b_), (0, 255, 0), 2)
                    cv2.putText(vis, f"ID:{t.track_id}", (l, max(15, t_ - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"⏸ Pauza na framu {frame_idx} – klikni na dron (levý klik) nebo klávesu pro auto-výběr")
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
                        print("❌ Klik nezasáhl žádný box – proveden automatický výběr")
                    click_pos = None

                # Pokud neklikneš nebo klik mimo → auto-výběr
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

        # 👉 vykreslujeme jen vybraný objekt (červeně)
        if vis is not None and tracks and selected_id is not None:
            for t in tracks:
                if not t.is_confirmed():
                    continue
                if t.track_id != selected_id:
                    continue

                tid = t.track_id
                l, t_, r, b_ = map(int, t.to_ltrb())
                cx = int((l + r) / 2)
                cy = int((t_ + b_) / 2)
                w_box = r - l
                h_box = b_ - t_

                color = (0, 0, 255)
                cv2.rectangle(vis, (l, t_), (r, b_), color, 2)
                cv2.putText(vis, f"ID:{tid}", (l, max(15, t_ - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # záznam do CSV
                csv_writer.writerow([frame_idx, tid, cx, cy, w_box, h_box, ""])

        # 👉 vykresli červený křížek v místě kliknutí (pro debug)
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
    parser.add_argument("--winsize", type=int, default=0,
                        help="velikost okna (šířka=výška), 0 = původní velikost")
    parser.add_argument("--pause_frame", type=int, default=0,
                        help="Frame číslo, kde se má video pozastavit pro ruční výběr (0 = hned na začátku)")
    args = parser.parse_args()
    main(args)
