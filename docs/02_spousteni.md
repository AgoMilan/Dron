# ▶️ Spouštění projektu Dron

### 🧠 Aktivace prostředí
```powershell
cd C:\Users\Milan\Projekty\Dron
.env\Scripts\activate
```

---
## 🚁 Tracker YOLO + DeepSORT
```powershell
cd src
python main.py --mode yolo --source test_videos/drone1.mp4 --show --pause_frame 15 --winsize 640 --model yolov8n.pt --yolo-conf 0.35 --max_age 55 --iou_dist 0.4 --n_init 3 --shrink 0.3
```

### Parametry
| Parametr | Popis |
|-----------|--------|
| `--mode yolo` | Aktivuje YOLOv8 detekci a DeepSORT tracking |
| `--source` | Cesta k testovacímu videu |
| `--show` | Zobrazí výstupní okno |
| `--pause_frame` | Pauza na daném snímku (pro test) |
| `--winsize` | Velikost zobrazení (640 px) |
| `--model` | Název YOLO modelu |
| `--yolo-conf` | Confidence threshold |
| `--max_age`, `--iou_dist`, `--n_init` | Parametry pro DeepSORT |
| `--shrink` | Měřítko výřezu |

---
## 🎯 Tracker základní s OpenCV
```powershell
python main.py --mode opencv --source test_videos\drone1.mp4 --show --winsize 640 --tracker MOSSE
```

### Podporované trackery
- MOSSE 🟢 (nejrychlejší, vhodný pro realtime testy)
- CSRT 🔵 (přesnější, ale pomalejší)
- KCF ⚪ (kompromis mezi rychlostí a přesností)
