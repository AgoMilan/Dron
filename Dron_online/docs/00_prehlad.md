# 💡 Projekt Dron_online – Přehled

### 🧭 Cíl
Automatické sledování objektů (dron, pták, osoba) pomocí YOLOv8 + DeepSORT
z RTSP kamery Vivotek s možností PTZ ovládání.

### ⚙️ Použité technologie
- Python 3.11 (venv)
- Ultralytics YOLOv8
- DeepSORT (deep_sort_realtime)
- OpenCV
- RTSP stream z Vivotek IP kamery
- Volitelně HTTP PTZ ovládání

### 📦 Struktura
| Složka | Popis |
|--------|--------|
| `src/` | Python skripty projektu |
| `runs/` | Uložené výstupy (video, CSV) |
| `models/` | Trénované YOLO modely |
| `docs/` | Dokumentace projektu |

### 📅 Aktuální stav
✅ Online stream z kamery Vivotek  
✅ Detekce a sledování objektů (YOLO + DeepSORT)  
✅ Zobrazení FPS a počtu objektů  
✅ Ovládání PTZ (ručně – šipky/WASD)  
🟡 Auto-tracking ve vývoji  
