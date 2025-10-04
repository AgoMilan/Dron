
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf=0.35, imgsz=640):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame):
        # runs model.predict and returns raw results
        results = self.model.predict(source=[frame], conf=self.conf, imgsz=self.imgsz,
                                     device=self.device, verbose=False)
        return results[0] if results else None
