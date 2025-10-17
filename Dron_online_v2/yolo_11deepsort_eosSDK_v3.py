# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 22:13:51 2025

@author: Milan
"""

# -*- coding: utf-8 -*-
"""
YOLOv11 + DeepSORT + Canon EOS77D (Live View přes EDSDK)
---------------------------------------------------------
Přímé snímání z kamery Canon EOS77D (USB, LiveView) pomocí EDSDK v13.19.10.
Zobrazuje YOLO detekce, FPS, latenci a loguje výkon do CSV.
Autor: Milan
"""

import argparse
import time
import cv2
import csv
import os
import numpy as np
import torch
import ctypes
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Cesta k Canon EDSDK knihovně
EDSDK_PATH = r"C:\Users\Milan\Projekty\Dron\EDSDKv131910W\EDSDKv131910W\Windows\EDSDK_64\Dll\EDSDK.dll"

# Canon EDSDK konstanty
EDS_OK = 0x00000000
kEdsPropID_Evf_OutputDevice = 0x00000500
kEdsEvfOutputDevice_PC = 0x00000002
kEdsCameraCommand_TakePicture = 0x00000000


# ===========================
# TŘÍDA PRO OVLÁDÁNÍ KAMERY
# ===========================
class CanonEOS77DLive:
    """Canon EOS 77D LiveView controller (EDSDK)."""

    def __init__(self):
        self.edsdk = ctypes.WinDLL(EDSDK_PATH)
        self.camera = ctypes.c_void_p()
        self.session_open = False

    def _check(self, err, action=""):
        if err != EDS_OK:
            raise RuntimeError(f"Canon SDK Error {hex(err)} při {action}")

    def initialize(self):
        """Inicializuje SDK a otevře session."""
        print("📸 Inicializuji Canon EDSDK...")
        self._check(self.edsdk.EdsInitializeSDK(), "EdsInitializeSDK")

        cam_list = ctypes.c_void_p()
        self._check(self.edsdk.EdsGetCameraList(ctypes.byref(cam_list)), "EdsGetCameraList")

        cam_ref = ctypes.c_void_p()
        self._check(self.edsdk.EdsGetChildAtIndex(cam_list, 0, ctypes.byref(cam_ref)), "EdsGetChildAtIndex")

        self.camera = cam_ref
        self.edsdk.EdsRelease(cam_list)
        self._check(self.edsdk.EdsOpenSession(self.camera), "EdsOpenSession")
        self.session_open = True

        # Zapnutí LiveView
        output_device = ctypes.c_int(kEdsEvfOutputDevice_PC)
        self._check(self.edsdk.EdsSetPropertyData(self.camera, kEdsPropID_Evf_OutputDevice,
                                                  0, ctypes.sizeof(output_device), ctypes.byref(output_device)),
                    "EdsSetPropertyData(Evf_OutputDevice)")
        print("✅ Live View zapnutý.")

    def get_liveview_frame(self):
        """Získá aktuální snímek z LiveView."""
        if not self.session_open:
            raise RuntimeError("Session není otevřena.")

        # Vytvoření EVF obrázku
        img_ref = ctypes.c_void_p()
        self._check(self.edsdk.EdsCreateMemoryStream(0, ctypes.byref(img_ref)), "EdsCreateMemoryStream")

        evf_image = ctypes.c_void_p()
        self._check(self.edsdk.EdsCreateEvfImageRef(img_ref, ctypes.byref(evf_image)), "EdsCreateEvfImageRef")

        # Stažení snímku z kamery
        err = self.edsdk.EdsDownloadEvfImage(self.camera, evf_image)
        if err != EDS_OK:
            self.edsdk.EdsRelease(evf_image)
            self.edsdk.EdsRelease(img_ref)
            return None

        # Získání ukazatele na paměťový blok
        pointer = ctypes.c_void_p()
        size = ctypes.c_uint64()
        self._check(self.edsdk.EdsGetPointer(img_ref, ctypes.byref(pointer)), "EdsGetPointer")
        self._check(self.edsdk.EdsGetLength(img_ref, ctypes.byref(size)), "EdsGetLength")

        # Převod na numpy (JPEG)
        buffer = (ctypes.c_ubyte * size.value).from_address(pointer.value)
        img_array = np.frombuffer(buffer, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Uvolnění objektů
        self.edsdk.EdsRelease(evf_image)
        self.edsdk.EdsRelease(img_ref)

        return frame

    def capture_photo(self):
        """Spustí závěrku."""
        if not self.session_open:
            raise RuntimeError("Session není otevřena.")
        print("📷 Pořizuji snímek...")
        self._check(self.edsdk.EdsSendCommand(self.camera, kEdsCameraCommand_TakePicture, 0), "TakePicture")

    def close(self):
        """Ukončí session a SDK."""
        if self.session_open:
            print("🔌 Zavírám session...")
            self.edsdk.EdsCloseSession(self.camera)
            self.session_open = False
        self.edsdk.EdsTerminateSDK()
        print("👋 Canon SDK ukončeno.")


# ===========================
# YOLO + DeepSORT
# ===========================
def extract_detections_from_result(res, frame_w, frame_h):
    dets = []
    if res is None or not hasattr(res, "boxes"):
        return dets

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy()

    for i, b in enumerate(xyxy):
        x1, y1, x2, y2 = b[:4]
        conf = float(confs[i]) if confs[i] is not None else 0.0
        cls_id = int(clss[i])
        cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
        dets.append(([x1, y1, x2, y2], conf, cls_name))
    return dets


# ===========================
# HLAVNÍ PROGRAM
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--winsize", type=int, default=1280)
    parser.add_argument("--shrink", type=float, default=0.6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.to(device)
    tracker = DeepSort(max_age=30, n_init=1, embedder="mobilenet")

    canon = CanonEOS77DLive()
    canon.initialize()

    log_file = "performance_log.csv"
    write_header = not os.path.exists(log_file)
    csv_file = open(log_file, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow(["timestamp", "fps", "latency_ms", "objects_detected"])

    cv2.namedWindow("Canon LiveView", cv2.WINDOW_NORMAL)
    frame_count = 0
    fps_display = 0
    last_time = time.time()

    while True:
        frame = canon.get_liveview_frame()
        if frame is None:
            continue

        start_time = time.time()
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.conf, verbose=False, device=device)
        dets = extract_detections_from_result(results[0], frame.shape[1], frame.shape[0]) if results else []
        tracks = tracker.update_tracks(dets, frame=frame)

        display = frame.copy()
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, tr.to_tlbr())
            
            det_conf = getattr(tr, 'det_conf', 0.0)
            if det_conf is None:
                det_conf = 0.0
            label = f"ID:{tr.track_id} {getattr(tr, 'det_class', '?')} {det_conf:.2f}"

            
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1
        elapsed = time.time() - last_time
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            last_time = time.time()

        latency_ms = (time.time() - start_time) * 1000
        overlay = f"{device.upper()} | FPS: {fps_display:.1f} | Latence: {latency_ms:.1f} ms | Objekty: {len(tracks)}"
        cv2.putText(display, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         round(fps_display, 2), round(latency_ms, 2), len(tracks)])
        csv_file.flush()

        cv2.imshow("Canon LiveView", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('p'):
            canon.capture_photo()

    csv_file.close()
    canon.close()
    cv2.destroyAllWindows()
    print(f"✅ Log uložen: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    main()
