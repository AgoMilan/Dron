# Nastavení cesty k projektu
$projectDir = "C:\Users\Milan\Projekty\Dron"
$venvDir = "$projectDir\venv"

Write-Host ">>> Vytvářím virtuální prostředí ve složce: $venvDir"

# Pokud venv už existuje, smažeme ho
if (Test-Path $venvDir) {
    Write-Host ">>> Odstraňuji staré virtuální prostředí..."
    Remove-Item -Recurse -Force $venvDir
}

# Vytvoření nového venv
python -m venv $venvDir

# Aktivace prostředí
Write-Host ">>> Aktivace prostředí..."
& "$venvDir\Scripts\Activate.ps1"

# Instalace balíčků
Write-Host ">>> Instalace balíčků z requirements.txt..."
pip install --upgrade pip
pip install -r "$projectDir\requirements.txt"

# Test YOLOv8
Write-Host ">>> Test YOLO predikce..."
python -m ultralytics predict model=yolov8n.pt source=https://ultralytics.com/images/bus.jpg device=cpu

Write-Host ">>> Hotovo! Výstupy najdeš v adresáři runs/predict"
