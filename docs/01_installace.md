# ⚙️ Instalace prostředí

### 💡 Požadavky
- Windows 10 nebo novější
- Python 3.11
- Virtuální prostředí `venv`

### 🧩 Postup instalace
```powershell
cd C:\Users\Milan\Projekty\Dron
python -m venv venv
.env\Scriptsctivate
pip install ultralytics opencv-python deep-sort-realtime numpy pandas
```

### 💾 Test prostředí
```powershell
python -c "import cv2, ultralytics; print('OK - prostředí funguje')"
```

### 📁 Umístění projektu
`C:\Users\Milan\Projekty\Dron\`
