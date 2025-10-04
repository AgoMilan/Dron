# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 22:25:49 2025

@author: Milan
"""

# src/multi_tracker.py
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort


def resize_frame(frame, winsize):
    if winsize is None:
        return frame
    h, w = frame.shape[:2]
    scale = winsize / w
    new_h = int(h * scale)
    return cv2.resize(frame, (winsize, new_h))


class MultiObjectTracker:
    def __init__(self, winsize=None):
        self.tracker = DeepSort(max_age=30)
        self.winsize = winsize

    def update(self, detections, frame):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def show(self, frame, window_name="Tracking"):
        """Zobrazí frame se správnou velikostí podle winsize."""
        display = resize_frame(frame, self.winsize)
        cv2.imshow(window_name, display)
