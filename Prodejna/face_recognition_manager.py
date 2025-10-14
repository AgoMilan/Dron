# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 11:02:21 2025

@author: Milan
"""

# face_recognition_manager.py
# -*- coding: utf-8 -*-
"""
FaceRecognitionManager
- jednoduchá třída pro výpočet face-embeddingů (facenet-pytorch),
  správa galerie (JSON), buffering embeddingů pro každý track a rozhodování
  známá / neznámá.
- metody: update_track(frame, track_id, box) -> (label_text, score)
- enroll_person(name, emb) pro ruční přidání do galerie
"""

import os
import json
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as T


class FaceRecognitionManager:
    def __init__(
        self,
        device='cpu',
        gallery_path="gallery.json",
        buffer_size=10,
        decision_min_samples=3,
        threshold=0.55,
        crop_margin=0.25,
    ):
        self.device = device
        self.gallery_path = gallery_path
        self.buffer_size = buffer_size
        self.decision_min_samples = decision_min_samples
        self.threshold = threshold
        self.crop_margin = crop_margin

        # embedder
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # transform pro facenet
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
        ])

        # galerie: dict name -> embedding (numpy)
        self.gallery = self._load_gallery()

        # per-track buffers: track_id -> deque of embeddings
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.track_last_seen = {}

        print(f"[FRM] Inicializováno (device={self.device}, gallery={len(self.gallery)} items)")

    # ---------------------------
    # Galerie
    # ---------------------------
    def _load_gallery(self):
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: np.array(v, dtype=float) for k, v in data.items()}
            except Exception as e:
                print(f"[FRM] Chyba při načítání galerie: {e}")
                return {}
        return {}

    def save_gallery(self):
        try:
            serial = {k: v.tolist() for k, v in self.gallery.items()}
            with open(self.gallery_path, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
            print(f"[FRM] Galerie uložena ({len(self.gallery)} položek) -> {self.gallery_path}")
        except Exception as e:
            print(f"[FRM] Chyba při ukládání galerie: {e}")

    def enroll_person(self, name, emb):
        """Uloží embedding (normalizovaný) do galerie pod jménem name."""
        emb = emb / np.linalg.norm(emb)
        self.gallery[name] = emb
        self.save_gallery()
        print(f"[FRM] Enrolled: {name}")

    # ---------------------------
    # Crop + embedding
    # ---------------------------
    def crop_face(self, frame, box):
        """Box: [x1,y1,x2,y2] ve pixelech (původní rozlišení). Vrací crop BGR."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * self.crop_margin)
        my = int(bh * self.crop_margin)
        x1n = max(0, x1 - mx)
        y1n = max(0, y1 - my)
        x2n = min(w, x2 + mx)
        y2n = min(h, y2 + my)
        if x2n - x1n <= 0 or y2n - y1n <= 0:
            return None
        return frame[y1n:y2n, x1n:x2n].copy()

    def get_embedding(self, face_crop):
        """Vrací L2-normalizovaný embedding (numpy)."""
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.embedder(t).cpu().numpy()[0]
            if np.linalg.norm(emb) == 0:
                return None
            emb = emb / np.linalg.norm(emb)
            return emb
        except Exception as e:
            print(f"[FRM] Chyba ve get_embedding: {e}")
            return None

    # ---------------------------
    # Matching
    # ---------------------------
    def match_embedding(self, emb):
        """Porovná s galerií. Vrátí (label, score) nebo (None, score)."""
        if emb is None or len(self.gallery) == 0:
            return None, 0.0
        best_label = None
        best_score = -1.0
        for label, g_emb in self.gallery.items():
            score = 1.0 - cosine(emb, g_emb)  # kosinová podobnost
            if score > best_score:
                best_score = score
                best_label = label
        if best_score >= self.threshold:
            return best_label, float(best_score)
        return None, float(best_score)

    # ---------------------------
    # Hlavní update pro track
    # ---------------------------
    def update_track(self, frame, track_id, box):
        """Zavolat v main loopu pro každý track.
        Vrátí (label_text, score).
        """
        face_crop = self.crop_face(frame, box)
        emb = self.get_embedding(face_crop)
        if emb is not None:
            self.track_buffers[track_id].append(emb)
            self.track_last_seen[track_id] = time.time()

        label_text = "unknown"
        score = 0.0

        if len(self.track_buffers[track_id]) >= self.decision_min_samples:
            avg_emb = np.mean(np.stack(list(self.track_buffers[track_id])), axis=0)
            if np.linalg.norm(avg_emb) > 0:
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                label, score = self.match_embedding(avg_emb)
                if label:
                    label_text = f"{label} ({score:.2f})"
                else:
                    label_text = f"unknown ({score:.2f})"
        return label_text, score

    # ---------------------------
    # Housekeeping
    # ---------------------------
    def cleanup(self, max_age=3.0):
        """Vyčistí buffery pro tracky, které zmizely."""
        now = time.time()
        to_delete = [tid for tid, ts in self.track_last_seen.items() if now - ts > max_age]
        for tid in to_delete:
            if tid in self.track_buffers:
                del self.track_buffers[tid]
            del self.track_last_seen[tid]
