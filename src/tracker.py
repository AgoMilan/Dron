# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:50:07 2025

@author: Milan
"""


import cv2
import os


def create_tracker(preferred="KCF"):
    """
    Vytvoří objekt trackeru podle zadané preference.
    Automaticky přepne na jiný, pokud vybraný není dostupný.
    """
    preferred = preferred.upper()
    order = [preferred] + [t for t in ["CSRT", "KCF", "MOSSE"] if t != preferred]

    for tracker_type in order:
        try:
            if tracker_type == "KCF":
                if hasattr(cv2, "TrackerKCF_create"):
                    print("➡️ Používám tracker: KCF")
                    return cv2.TrackerKCF_create()
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
                    print("➡️ Používám tracker: legacy.KCF")
                    return cv2.legacy.TrackerKCF_create()

            elif tracker_type == "CSRT":
                if hasattr(cv2, "TrackerCSRT_create"):
                    print("➡️ Používám tracker: CSRT")
                    return cv2.TrackerCSRT_create()
                elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                    print("➡️ Používám tracker: legacy.CSRT")
                    return cv2.legacy.TrackerCSRT_create()

            elif tracker_type == "MOSSE":
                if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                    print("➡️ Používám tracker: legacy.MOSSE")
                    return cv2.legacy.TrackerMOSSE_create()
                elif hasattr(cv2, "TrackerMOSSE_create"):
                    print("➡️ Používám tracker: MOSSE")
                    return cv2.TrackerMOSSE_create()

        except Exception as e:
            print(f"⚠️ Tracker {tracker_type} selhal: {e}")

    raise RuntimeError("❌ Nepodařilo se vytvořit žádný dostupný tracker (KCF/CSRT/MOSSE).")


class DroneTrackerApp:
    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Nelze otevřít video: {args.source}")

        # výstupní video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

        self.tracker = None
        self.roi = None
        self.scale = 1.0  # měřítko videa při zmenšení (winsize)

    def run(self):
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_idx += 1

            # Přepočítání velikosti podle --winsize (zachová poměr stran)
            if self.args.winsize:
                h, w = frame.shape[:2]
                self.scale = self.args.winsize / w
                frame = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))

            # Pauza na výběr ROI
            if frame_idx == self.args.pause_frame:
                print("⏸ Pauza – vyber objekt myší a potvrď ENTER (ESC pro zrušení).")
                r = cv2.selectROI("Výběr objektu", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Výběr objektu")
                print("Vybrané ROI:", r)

                # Přetypování na int
                x, y, w, h = map(int, r)

                if w > 0 and h > 0:
                    try:
                        self.tracker = create_tracker(self.args.tracker)
                        frame_for_tracker = frame.copy()
                        ok = self.tracker.init(frame_for_tracker, (x, y, w, h))

                        if not ok:
                            print("⚠️ Tracker se nepodařilo inicializovat – nízký kontrast nebo mimo rám.")
                            return
                        else:
                            print("✅ Tracker inicializován s ROI:", (x, y, w, h))
                    except Exception as e:
                        print("❌ Chyba při inicializaci trackeru:", e)
                        return
                else:
                    print("⚠️ Nebyl vybrán žádný objekt.")
                    return

            # update trackeru
            if self.tracker is not None:
                ok, bbox = self.tracker.update(frame)
                if ok:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Ztraceno", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # zapisujeme do výstupu (originální velikost, neškálovaná)
            self.out.write(frame)

            if self.args.show:
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC pro ukončení
                    break

        print("✅ Hotovo – video dokončeno.")
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
