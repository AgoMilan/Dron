# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 12:46:15 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
Hlavní spouštěcí skript pro projekt DRON
Podporuje:
 - OpenCV trackery (KCF, CSRT, MOSSE)
 - YOLOv8 + DeepSORT sledování s výběrem tříd
Autor: Milan + GPT-5
"""

import argparse
import cv2
import os
import time


# =====================================================
# ============= OpenCV tracker režim ==================
# =====================================================
def run_opencv_tracker(args):
    print("▶️ Spouštím OpenCV tracker...")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Nelze otevřít zdroj videa: {args.source}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Nelze načíst první snímek.")

    # === Zmenšení okna pro zobrazení ===
    orig_h, orig_w = frame.shape[:2]
    scale = args.winsize / orig_w
    new_w = args.winsize
    new_h = int(orig_h * scale)
    print(f"Škála: {scale:.4f} -> zobrazím {new_w}x{new_h}")

    # === Výběr objektu ===
    if args.show:
        resized_frame = cv2.resize(frame, (new_w, new_h))
        print("⏸ Pauza – vyber objekt myší a potvrď ENTER (ESC pro zrušení).")
        bbox_scaled = cv2.selectROI("Výběr objektu", resized_frame, False)
        cv2.destroyWindow("Výběr objektu")

        # Přepočet vybraného ROI zpět do originální velikosti
        bbox = (
            bbox_scaled[0] / scale,
            bbox_scaled[1] / scale,
            bbox_scaled[2] / scale,
            bbox_scaled[3] / scale
        )
    else:
        bbox = (100, 100, 100, 100)

    print(f"Vybrané ROI (orig): {bbox}")

    # Detekce správného API (OpenCV vs. legacy)
    if hasattr(cv2, "legacy"):
        tracker_types = {
            "KCF": cv2.legacy.TrackerKCF_create,
            "CSRT": cv2.legacy.TrackerCSRT_create,
            "MOSSE": cv2.legacy.TrackerMOSSE_create
            }
    else:
       tracker_types = {
           "KCF": cv2.TrackerKCF_create,
           "CSRT": cv2.TrackerCSRT_create,
           "MOSSE": cv2.TrackerMOSSE_create
           }


    tracker = tracker_types[args.tracker]()
    ok = tracker.init(frame, bbox)
    if not ok:
        print("⚠️ Tracker se nepodařilo inicializovat – nízký kontrast nebo mimo rám.")
        return

    frame_idx = 0
    start_time = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        ok, bbox = tracker.update(frame)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "ZTRACENO", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Zmenšení pro zobrazení
        if args.show:
            display = cv2.resize(frame, (new_w, new_h))
            cv2.imshow("Tracking", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    dt = time.time() - start_time
    print(f"✅ Hotovo — frames: {frame_idx}, čas: {dt:.1f}s, FPS: {frame_idx/dt:.2f}")
    cap.release()
    cv2.destroyAllWindows()



# =====================================================
# ============= YOLO + DeepSORT režim =================
# =====================================================
def run_yolo_tracker(args):
    print("▶️ Spouštím YOLO + DeepSORT tracker...")

    # Aliasy pro kompatibilitu se starším skriptem
    if not hasattr(args, "display"):
        args.display = args.show
    if not hasattr(args, "csv"):
        args.csv = "runs/out.csv"
    if not hasattr(args, "classes"):
        args.classes = None

    import yolo_deepsort_tracker as ydt
    ydt.main(args)


# =====================================================
# ============= Argumenty programu ====================
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Sledování dronu – hlavní spouštěcí skript")

    # Společné argumenty
    parser.add_argument("--mode", choices=["opencv", "yolo"], required=True,
                        help="Režim sledování: 'opencv' nebo 'yolo'")
    parser.add_argument("--source", type=str, required=True, help="Vstupní video nebo kamera")
    parser.add_argument("--output", type=str, default="runs/out.mp4", help="Výstupní video")
    parser.add_argument("--show", action="store_true", help="Zobrazit video v reálném čase")
    parser.add_argument("--pause_frame", type=int, default=15,
                        help="Frame pro pozastavení a výběr objektu")
    parser.add_argument("--winsize", type=int, default=640,
                        help="Velikost zobrazovacího okna")
    parser.add_argument("--tracker", choices=["KCF", "CSRT", "MOSSE"], default="KCF",
                        help="Typ OpenCV trackeru")

    # ⚙️ Parametry pro YOLO režim (kompatibilní s track_drone23.py)
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model (např. yolov8n.pt, yolov8m.pt, yolov8x.pt)")
    parser.add_argument("--yolo-conf", dest="conf", type=float, default=0.35,
                        help="Práh důvěry detekce pro YOLO")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Filtrované třídy objektů např. drone airplane bird car bus (nebo 'all')")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Velikost vstupního obrazu pro YOLO")
    parser.add_argument("--max_age", type=int, default=55,
                        help="Počet snímků, po které se track udrží i bez detekce")
    parser.add_argument("--iou_dist", type=float, default=0.4,
                        help="Maximální IoU vzdálenost pro propojení tracků")
    parser.add_argument("--n_init", type=int, default=3,
                        help="Počet nutných detekcí k potvrzení tracku")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Zařízení: 'cpu' nebo 'cuda'")
    parser.add_argument("--csv", type=str, default="runs/out.csv",
                        help="CSV soubor pro logování výsledků (YOLO režim)")
    parser.add_argument("--shrink", type=float, default=0.25,
                    help="Poměr zmenšení ohraničujícího rámečku (0.0–0.9, výchozí 0.25)")


    return parser.parse_args()


# =====================================================
# ================== Spuštění =========================
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    if args.mode == "opencv":
        run_opencv_tracker(args)
    elif args.mode == "yolo":
        run_yolo_tracker(args)
