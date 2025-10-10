# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:15:03 2025

@author: Milan
"""

import cv2
import time

# Zkus otevřít všechna možná video zařízení (0–3)
for index in range(4):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"✅ Kamera detekována na indexu {index}")
        print(f"  - Rozlišení: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        cap.release()
    else:
        print(f"❌ Kamera na indexu {index} není dostupná")

print("\nTeď zkusíme zobrazit živý stream z první aktivní kamery...")
time.sleep(2)

# Znovu otevři první aktivní zařízení
for index in range(4):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"\n🎥 Zkouším číst z kamery index {index}")
        ok, frame = cap.read()
        if ok:
            print(f"✅ Frame přijat, typ dat: {type(frame)}, rozměr: {frame.shape}")
            cv2.imshow("EOS Diagnostic", frame)
            cv2.waitKey(2000)  # zobrazí okno na 2 sekundy
            cv2.destroyAllWindows()
            cap.release()
            break
        else:
            print(f"⚠️ Kamera {index} je otevřená, ale nevrací žádný obraz (pravděpodobně logo nebo placeholder).")
        cap.release()

print("\n✅ Hotovo. Pokud žádné okno neukázalo obraz, EOS Utility zatím neposílá video.")
