# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 16:00:58 2025

@author: Milan
"""

import cv2

class VideoCamera:
    def __init__(self, source):
        # Pokus o otevření zdroje videa (soubor nebo kamera)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

    def read(self):
        """Vrátí další snímek z videa jako numpy.ndarray.
        Pokud není k dispozici, vrací None.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame  # ✅ vždy jen samotný obrázek, ne tuple

    def release(self):
        """Uvolní prostředky"""
        if self.cap:
            self.cap.release()
