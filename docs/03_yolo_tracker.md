# 🧠 YOLO + DeepSORT – sledování dronu

### Popis režimu
YOLOv8 provádí detekci objektů a DeepSORT zajišťuje jejich sledování v čase.  
Každý objekt má unikátní ID, které se nemění mezi snímky.

### Příklad příkazu
```powershell
python main.py --mode yolo --source test_videos/drone1.mp4 --show --winsize 640
```

### Výhody
✅ Stabilní sledování pohybujících se objektů  
✅ Vhodné pro budoucí napojení na online stream  
✅ Dobrá přesnost i u menších objektů  
