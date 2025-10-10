# 💡 Projekt Dron – Přehled

### 🧭 Cíl projektu
Offline testování a ladění algoritmů pro sledování objektů (např. dronů) na video souborech.  
Projekt využívá YOLOv8 a OpenCV trackery pro analýzu pohybu.

### ⚙️ Použité technologie
- Python 3.11 (venv)
- Ultralytics YOLOv8
- OpenCV trackery (MOSSE, CSRT, KCF)
- DeepSORT (pro YOLO režim)

### 📦 Struktura projektu
| Složka | Popis |
|--------|--------|
| `src/` | Zdrojové Python skripty |
| `test_videos/` | Testovací videa (.mp4) |
| `runs/` | Výstupy (uložené video, CSV logy) |
| `docs/` | Dokumentace projektu |

### 📅 Aktuální stav
✅ Testovací videa fungují  
✅ YOLO + DeepSORT detekce běží  
✅ OpenCV trackery MOSSE a CSRT testovány  
🟡 Integrace s online kamerou ve vývoji  
