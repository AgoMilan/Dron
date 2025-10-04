# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 18:59:38 2025

@author: Milan
"""
import torch
import ultralytics
from ultralytics import YOLO

print("✅ Test GPU prostředí")
print("="*40)

# Verze
print(f"PyTorch verze: {torch.__version__}")
print(f"Ultralytics YOLO verze: {ultralytics.__version__}")

# GPU kontrola
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    print(f"🟢 CUDA dostupná")
    print(f"   GPU: {gpu_name}")
    print(f"   Architektura (sm_xx): sm_{gpu_capability[0]}{gpu_capability[1]}")
else:
    print("🔴 CUDA není dostupná (běží jen CPU)")

# Test YOLO
print("\n>>> Test YOLO predikce...")
try:
    model = YOLO("yolov8n.pt")
    results = model.predict(
        source="https://ultralytics.com/images/bus.jpg",
        device=0 if torch.cuda.is_available() else "cpu",
        save=True
    )
    print("✅ YOLO predikce proběhla, výstup je v runs/predict/")
except Exception as e:
    print("❌ Chyba při běhu YOLO:", e)
