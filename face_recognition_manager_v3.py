# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:46:53 2025

@author: Milan

FaceRecognitionManager v4.1 (2025-10)
------------------------------------
- Správa embeddingů a galerie osob (gallery.json)
- Práce s návštěvníky přes VisitorDB
- Automatické přidání neznámých osob po několika výskytech
- Nově: ukládání náhledů obličejů při přidání nové osoby
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
        decision_min_samples=2,
        threshold=0.65,
        crop_margin=0.25,
        visitor_db=None,
        auto_add_threshold=2,  # kolikrát musí být unknown, než se přidá nový visitor
        thumbnails_dir="thumbnails"
    ):
        self.device = device
        self.gallery_path = gallery_path
        self.buffer_size = buffer_size
        self.decision_min_samples = decision_min_samples
        self.threshold = threshold
        self.crop_margin = crop_margin
        self.visitor_db = visitor_db
        self.auto_add_threshold = auto_add_threshold
        self.thumbnails_dir = thumbnails_dir

        # inicializace modelu pro výpočet embeddingu (InceptionResnet)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
        ])

        # načtení galerie známých osob
        self.gallery = self._load_gallery()
        # paměť pro jednotlivé tracky (krátkodobé ukládání embeddingů)
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.track_last_seen = {}
        self.track_unknown_counts = defaultdict(int)

        os.makedirs(self.thumbnails_dir, exist_ok=True)

        print(f"[FRMv4.1] Inicializováno (device={self.device}, gallery={len(self.gallery)} osob)")

    # --------------------------- Galerie ---------------------------
    def _load_gallery(self):
        """Načte známé osoby z JSON souboru."""
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: [np.array(e, dtype=float) for e in v] for k, v in data.items()}
            except Exception as e:
                print(f"[FRM] Chyba při načítání galerie: {e}")
        return {}

    def save_gallery(self):
        """Uloží aktuální galerii do JSON."""
        try:
            serial = {k: [e.tolist() for e in v] for k, v in self.gallery.items()}
            with open(self.gallery_path, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[FRM] Chyba při ukládání galerie: {e}")

    def enroll_person(self, name, emb, face_crop=None):
        """Přidá novou známou osobu do galerie a uloží náhled."""
        if emb is None:
            print("[FRM] Nelze přidat osobu: embedding je None.")
            return
        emb = np.array(emb).reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        if name not in self.gallery:
            self.gallery[name] = []
        self.gallery[name].append(emb)
        self.save_gallery()

        # uložit náhledový obrázek
        if face_crop is not None:
            thumb_path = os.path.join(self.thumbnails_dir, f"{name}.jpg")
            cv2.imwrite(thumb_path, face_crop)
            print(f"[FRM] Náhled uložen: {thumb_path}")

        print(f"[FRM] Nová osoba přidána do galerie: {name}")

    def delete_person(self, name):
        """Odstraní osobu z galerie."""
        if name in self.gallery:
            del self.gallery[name]
            self.save_gallery()
            return True
        return False

    # --------------------------- Embedding funkce ---------------------------
    def crop_face(self, frame, box):
        """Ořeže oblast obličeje s mírným rozšířením."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        mx, my = int(bw * self.crop_margin), int(bh * self.crop_margin)
        x1n, y1n = max(0, x1 - mx), max(0, y1 - my)
        x2n, y2n = min(w, x2 + mx), min(h, y2 + my)
        if x2n - x1n <= 20 or y2n - y1n <= 20:
            return None
        return frame[y1n:y2n, x1n:x2n].copy()

    def get_embedding(self, face_crop):
        """Získá embedding (vektorový otisk) z oříznutého obličeje."""
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] < 20 or rgb.shape[1] < 20:
                return None
            t = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.embedder(t).cpu().numpy()
            emb = emb[0] if emb.ndim == 2 else emb
            return emb / (np.linalg.norm(emb) + 1e-12)
        except Exception as e:
            print(f"[FRM] Chyba ve get_embedding: {e}")
            return None

    def match_gallery(self, emb):
        """Porovná embedding s galerií známých osob."""
        if emb is None or len(self.gallery) == 0:
            return None, 0.0
        best_label, best_score = None, -1.0
        for label, elist in self.gallery.items():
            for e in elist:
                score = 1.0 - cosine(emb, e)
                if score > best_score:
                    best_label, best_score = label, score
        if best_score >= self.threshold:
            return best_label, float(best_score)
        return None, float(best_score)

    # --------------------------- Hlavní rozpoznávání ---------------------------
    def update_track(self, frame, track_id, box):
        """Zpracuje detekovaný obličej, vrátí label, skóre a info o návštěvníkovi."""
        face_crop = self.crop_face(frame, box)
        emb = self.get_embedding(face_crop)
        if emb is not None and emb.shape[0] > 10:
            self.track_buffers[track_id].append(emb)
            self.track_last_seen[track_id] = time.time()

        label_text = "unknown"
        score = 0.0
        visitor_info = None

        if len(self.track_buffers[track_id]) >= self.decision_min_samples:
            try:
                # průměrný embedding z několika snímků
                buf = np.stack(list(self.track_buffers[track_id]), axis=0)
                avg_emb = np.mean(buf, axis=0)
                avg_emb /= (np.linalg.norm(avg_emb) + 1e-6)

                # 1️⃣ pokus o nalezení v galerii známých osob
                name, gscore = self.match_gallery(avg_emb)
                if name:
                    label_text = f"{name} ({gscore:.2f})"
                    self.track_unknown_counts[track_id] = 0
                    return label_text, gscore, None

                # 2️⃣ pokus o nalezení v databázi návštěvníků
                if self.visitor_db is not None:
                    vid, vscore = self.visitor_db.find_match(avg_emb)
                    if vid:
                        self.visitor_db.update_existing(vid, avg_emb)
                        rec = self.visitor_db.get_record(vid)
                        label_text = f"{vid} (visits:{rec['visits']})"
                        visitor_info = {"id": vid, "visits": rec["visits"], "score": vscore}
                        self.track_unknown_counts[track_id] = 0
                        return label_text, vscore, visitor_info

                # 3️⃣ pokud je obličej stále neznámý, po několika výskytech se přidá nový
                self.track_unknown_counts[track_id] += 1
                cnt = self.track_unknown_counts[track_id]
                if self.visitor_db is not None and cnt >= self.auto_add_threshold:
                    new_vid = self.visitor_db.add_new(avg_emb)
                    rec = self.visitor_db.get_record(new_vid)
                    label_text = f"{new_vid} (visits:{rec['visits']})"
                    visitor_info = {"id": new_vid, "visits": rec["visits"], "score": None}
                    self.track_unknown_counts[track_id] = 0

                    # uloží náhled obličeje
                    if face_crop is not None:
                        cv2.imwrite(os.path.join(self.thumbnails_dir, f"{new_vid}.jpg"), face_crop)

                    print(f"[FRM] ✅ Nový návštěvník přidán: {new_vid}")
                    return label_text, 0.0, visitor_info

            except Exception as e:
                print(f"[FRM] Chyba při vyhodnocení: {e}")

        return label_text, score, None

    def cleanup(self, max_age=3.0):
        """Odstraní staré tracky, které dlouho nebyly vidět."""
        now = time.time()
        to_del = [tid for tid, ts in self.track_last_seen.items() if now - ts > max_age]
        for tid in to_del:
            self.track_buffers.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_unknown_counts.pop(tid, None)
