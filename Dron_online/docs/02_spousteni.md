# ▶️ Spouštění projektu

### 📸 Spuštění online sledování
```powershell
python .\yolo_deepsort_online_vivotek_v2.py --source "rtsp://root:asd456@192.168.0.205:554/live.sdp" --display
```

### 🎛️ Spuštění s PTZ ovládáním
```powershell
python .\yolo_deepsort_online_vivotek_v2.py --source "rtsp://root:asd456@192.168.0.205:554/live.sdp" --display --ptz_url "http://root:asd456@192.168.0.205/cgi-bin/camctrl/ptz.cgi?move={cmd}"
```

### 💻 Lokální kamera (test)
```powershell
python .\yolo_deepsort_online_vivotek_v2.py --source 0 --display
```
