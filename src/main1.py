import argparse
import pathlib
import cv2

from cameras import VideoCamera
from detector import Detector
from tracker import Tracker
from gimbal import GimbalController
from logger import CSVLogger
import yaml

class DroneTrackerApp:
    def __init__(self, args):
        self.args = args
        # Kamera
        self.cam = VideoCamera(args.source)
        # Detektor YOLO
        self.detector = Detector(model_path=args.model, device=args.device,
                                 conf=args.conf, imgsz=args.imgsz)
        # Tracker DeepSort
        self.tracker = Tracker(max_age=args.max_age, n_init=args.n_init,
                               max_iou_distance=args.iou_dist,
                               max_cosine_distance=args.max_cosine_distance,
                               nn_budget=args.nn_budget, embedder=args.embedder,
                               half=args.half, bgr=args.bgr)
        # Stub gimbalu
        self.gimbal = GimbalController()

        # Logger
        out_csv = pathlib.Path(args.output).with_suffix(".csv")
        self.logger = CSVLogger(str(out_csv))
        self.frame_idx = 0

        # VideoWriter pro výstup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(str(args.output), fourcc, self.cam.fps,
                                      (self.cam.width, self.cam.height))

    def run(self):
        print("▶️ Running app (press 'q' to quit)")
        while True:
            frame, ts = self.cam.read()
            if frame is None:
                break

            # --- Detekce ---
            res = self.detector.detect(frame)
            dets = []
            if res and res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for i, b in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, b[:4])
                    w = x2 - x1
                    h = y2 - y1
                    dets.append(([x1, y1, w, h], float(confs[i]), int(clss[i])))

            # --- Tracking ---
            tracks = self.tracker.update(dets, frame=frame)

            # --- Vykreslení a logování ---
            if tracks:
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    l, t_, r, b_ = map(int, t.to_ltrb())
                    cx = int((l + r) / 2)
                    cy = int((t_ + b_) / 2)
                    w_box = r - l
                    h_box = b_ - t_
                    state = "detekce" if t.is_confirmed() and t.time_since_update == 0 else "predikce"

                    # Box + ID
                    color = (0, 255, 0) if state == "detekce" else (0, 255, 255)
                    cv2.rectangle(frame, (l, t_), (r, b_), color, 2)
                    cv2.putText(frame, f"ID {t.track_id}", (l, t_ - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Log do CSV
                    self.logger.log([self.frame_idx, t.track_id, cx, cy,
                                     w_box, h_box, state, None, None])

            # --- Zápis do videa ---
            self.writer.write(frame)

            # --- Zobrazení (--show + --winsize) ---
            if self.args.show:
                disp_frame = frame.copy()
                if self.args.winsize and self.args.winsize > 0:
                    disp_frame = cv2.resize(disp_frame, (self.args.winsize, self.args.winsize))
                cv2.imshow("Tracking", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # --- Pauza (--pause_frame) ---
            if self.args.pause_frame and self.frame_idx == self.args.pause_frame:
                print(f"⏸ Pauza na frame {self.frame_idx}, stiskni klávesu pro pokračování...")
                cv2.waitKey(0)

            self.frame_idx += 1

        # --- Cleanup ---
        self.logger.close()
        self.writer.release()
        cv2.destroyAllWindows()
        print("Done.")


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=None,
               help="YAML config file with parameters")
    p.add_argument('--source', type=str, default=0,
                   help="Video file path or camera index")
    p.add_argument('--output', type=str, default='runs/track_out.mp4',
                   help="Output video file")
    p.add_argument('--model', type=str, default='yolov8n.pt')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', type=str, default='cpu')

    # --- nové parametry kompatibilní s track_drone22.py ---
    p.add_argument('--show', action='store_true',
                   help="Show video window while processing")
    p.add_argument('--winsize', type=int, default=640,
                   help="Resize display window size")
    p.add_argument('--pause_frame', type=int, default=None,
                   help="Pause after N frames for debug/target selection")

    # deep sort args
    p.add_argument('--max_age', type=int, default=60)
    p.add_argument('--n_init', type=int, default=3)
    p.add_argument('--iou_dist', type=float, default=0.7)
    p.add_argument('--max_cosine_distance', type=float, default=0.2)
    p.add_argument('--nn_budget', type=int, default=None)
    p.add_argument('--embedder', type=str, default='mobilenet')
    p.add_argument('--half', action='store_true')
    p.add_argument('--bgr', action='store_true')
    return p


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    # přepíšeme hodnoty z configu do args
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    app = DroneTrackerApp(args)
    app.run()
