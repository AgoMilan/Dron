# 🎯 OpenCV Trackery – základní sledování

### Popis režimu
Režim `opencv` používá nativní OpenCV trackery.  
Každý běží bez nutnosti detekce – ručně označíš objekt a systém ho sleduje.

### Příklad spuštění
```powershell
python main.py --mode opencv --source test_videos\drone1.mp4 --show --winsize 640 --tracker MOSSE
```

### Typy trackerů
| Tracker | Popis |
|----------|--------|
| MOSSE | Rychlý, nízká přesnost |
| CSRT | Přesný, pomalejší |
| KCF | Kompromis |

### Doporučení
Pro rychlé testy použij `MOSSE`, pro finální výstup `CSRT`.
