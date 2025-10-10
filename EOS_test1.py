# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:37:32 2025

@author: Milan
"""

import cv2

cap = cv2.VideoCapture(0)  # nebo jiný index, pokud máš víc kamer

if not cap.isOpened():
    print("❌ Kamera není otevřena.")
else:
    print("✅ Kamera připojena, stiskni Q pro ukončení.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame nelze načíst.")
            break
        cv2.imshow("EOS Live Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
