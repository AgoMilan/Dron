# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 22:10:35 2025

@author: Milan
"""

# face_recognition_manager_v2_7.py
# -*- coding: utf-8 -*-
"""
FaceRecognitionManager v4
- Více embeddingů / osoba (gallery.json)
- Spolupráce s VisitorDB (visitors.json) pro automatickou evidenci unknown visitorů
- Automatické přidání nového visitora po `auto_add_threshold` nezávislých výskytů tracku
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

# import visitor db if present (path should be provided by caller)
# from visitor_db import VisitorDB  <-- caller will create instance and pass to manager


class FaceRecognitionManager:
    def __init__(
        self,
        device='cpu',
        gallery_path="gallery.json",
        buffer_size=10,
        decision_min_samples=3,
        threshold=0.65,
        crop_margin=0.25,
        visitor_db=None,
        auto_add_threshold=2,   # kolikrát musí track být detekován jako unknown než se založí visitor
    ):
        self.device = device
        self.gallery_path = gallery_path
        self.buffer_size = buffer_size
        self.decision_min_samples = decision_min_samples
        self.threshold = threshold
        self.crop_margin = crop_margin
        self.visitor_db = visitor_db
        self.auto_add_threshold = auto_add_threshold

        # model
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((160, 160)),
            T.ToTensor(),
        ])

        # gallery: person -> list of embeddings (numpy arrays)
        self.gallery = self._load_gallery()
        self.track_buffers = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self.track_last_seen = {}
        # track counters for unknown auto-add
        self.track_unknown_counts = defaultdict(int)

        print(f"[FRMv4] Inicializováno (device={self.device}, gallery={len(self.gallery)} persons, visitor_db={'yes' if self.visitor_db else 'no'})")

    # ---------------------------
    # Gallery
    # ---------------------------
    def _load_gallery(self):
        if os.path.exists(self.gallery_path):
            try:
                with open(self.gallery_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: [np.array(e, dtype=float) for e in v] for k, v in data.items()}
            except Exception as e:
                print(f"[FRMv4] Chyba při načítání galerie: {e}")
                return {}
        return {}

    def save_gallery(self):
        try:
            serial = {k: [e.tolist() for e in v] for k, v in self.gallery.items()}
            with open(self.gallery_path, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
            #print(f"[FRMv4] Gallery saved -> {self.gallery_path}")
        except Exception as e:
            print(f"[FRMv4] Chyba při ukládání galerie: {e}")

    def enroll_person(self, name, emb):
        """Přidá embedding do gallery pod jménem (nový nebo existující)."""
        if emb is None:
            print("[FRMv4] Nelze enrollovat: embedding je None.")
            return
        emb = np.array(emb).reshape(-1)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        if name not in self.gallery:
            self.gallery[name] = []
        self.gallery[name].append(emb)
        self.save_gallery()
        print(f"[FRMv4] Enrolled sample for: {name} (total {len(self.gallery[name])})")

    def delete_person(self, name):
        if name in self.gallery:
            del self.gallery[name]
            self.save_gallery()
            return True
        return False

    def list_gallery(self):
        return sorted(list(self.gallery.keys()))

    # ---------------------------
    # Embedding utils
    # ---------------------------
    def crop_face(self, frame, box):
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
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] < 20 or rgb.shape[1] < 20:
                return None
            t = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.embedder(t).cpu().numpy()
            if emb.ndim == 2:
                emb = emb[0]
            emb = emb.reshape(-1)
            if np.linalg.norm(emb) == 0:
                return None
            return emb / np.linalg.norm(emb)
        except Exception as e:
            print(f"[FRMv4] Chyba ve get_embedding: {e}")
            return None

    def match_gallery(self, emb):
        """Najdi nejlepší shodu v gallery. Vrátí (name,score) nebo (None,0)."""
        if emb is None or len(self.gallery) == 0:
            return None, 0.0
        emb = np.array(emb).reshape(-1)
        best_label, best_score = None, -1.0
        for label, elist in self.gallery.items():
            for e in elist:
                e = np.array(e).reshape(-1)
                if e.shape != emb.shape:
                    continue
                score = 1.0 - cosine(emb, e)
                if score > best_score:
                    best_label, best_score = label, score
        if best_score >= self.threshold:
            return best_label, float(best_score)
        return None, float(best_score)

    # ---------------------------
    # Hlavní update (volá se z hlavní smyčky)
    # ---------------------------
    def update_track(self, frame, track_id, box):
        """
        Vrátí (label_text, score, visitor_info)
        visitor_info je dict pokud visitor rozpoznán/created
        """
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
                buf = np.stack(list(self.track_buffers[track_id]), axis=0)
                avg_emb = np.mean(buf, axis=0).reshape(-1)
                avg_emb /= (np.linalg.norm(avg_emb) + 1e-6)

                # 1) check gallery (known persons)
                name, gscore = self.match_gallery(avg_emb)
                if name:
                    label_text = f"{name} ({gscore:.2f})"
                    # reset unknown counter
                    self.track_unknown_counts[track_id] = 0
                    return label_text, gscore, None

                # 2) check visitor DB (unknowns)
                if self.visitor_db is not None:
                    vid, vscore = self.visitor_db.find_match(avg_emb)
                    if vid:
                        # existing visitor found -> update
                        self.visitor_db.update_existing(vid, avg_emb)
                        rec = self.visitor_db.get_record(vid)
                        label_text = f"{vid} (visits:{rec['visits']})"
                        visitor_info = {"id": vid, "visits": rec["visits"], "score": vscore}
                        # reset unknown counter
                        self.track_unknown_counts[track_id] = 0
                        return label_text, vscore, visitor_info

                # 3) unknown -> increment unknown counter for this track
                self.track_unknown_counts[track_id] += 1
                cnt = self.track_unknown_counts[track_id]

                # if seen enough times as unknown, create new visitor record
                if self.visitor_db is not None and cnt >= self.auto_add_threshold:
                    new_vid = self.visitor_db.add_new(avg_emb)
                    rec = self.visitor_db.get_record(new_vid)
                    label_text = f"{new_vid} (visits:{rec['visits']})"
                    visitor_info = {"id": new_vid, "visits": rec["visits"], "score": None}
                    # reset counter for this track
                    self.track_unknown_counts[track_id] = 0
                    return label_text, 0.0, visitor_info

                # otherwise remain unknown, provide score from best visitor candidate for info
                # try to find best visitor candidate score (without auto update)
                if self.visitor_db is not None:
                    # compute best score (for UI only)
                    best_vid, best_vs = self.visitor_db.find_match(avg_emb)
                    if best_vid:
                        score = best_vs
                    else:
                        score = 0.0
                else:
                    score = 0.0

                label_text = f"unknown ({score:.2f})"
            except Exception as e:
                print(f"[FRMv4] Chyba při vyhodnocení: {e}")

        return label_text, score, visitor_info

    def cleanup(self, max_age=3.0):
        now = time.time()
        to_del = [tid for tid, ts in self.track_last_seen.items() if now - ts > max_age]
        for tid in to_del:
            self.track_buffers.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_unknown_counts.pop(tid, None)
