
import cv2
import time

class VideoCamera:
    """Simple wrapper for cv2.VideoCapture that returns frames with timestamps."""
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ret, frame = self.cap.read()
        ts = time.time()
        if not ret:
            return None, ts
        return frame, ts

    def release(self):
        self.cap.release()
