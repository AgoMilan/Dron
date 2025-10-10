# 🧠 YOLO + DeepSORT – sledování objektů

### YOLOv8
- Rychlý model detekce objektů
- Použitý model: `yolov8n.pt`
- Konfidence: `--conf 0.35`
- Vstupní velikost: `--imgsz 640`

### DeepSORT
- Sleduje objekty napříč snímky
- Každý objekt má vlastní ID (např. `ID:2 person`)
- Použitý embedder: `mobilenet`

### Výstup
- Zelený obdélník kolem detekce
- FPS a počet objektů v obraze
