# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2

# index zařízení může být 0,1,2 podle systému; zkontroluj, kde je EOS Webcam Utility
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Nelze otevřít video zařízení")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nelze číst snímek")
        break
    cv2.imshow("EOS Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC klávesa ukončí
        break

cap.release()
cv2.destroyAllWindows()

