# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 22:24:58 2025

@author: Milan
"""

# src/detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # ✅ DeepSORT formát: ([x1,y1,x2,y2], confidence, class_id)
                detections.append(([x1, y1, x2, y2], conf, cls))
        return detections
