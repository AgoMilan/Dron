# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 22:09:34 2025

@author: Milan
"""

# visitor_db.py
# -*- coding: utf-8 -*-
"""
VisitorDB - automatická evidence návštěvníků (unknown_X)
- Ukládá visitors.json s položkami:
  visitor_id: {
    "embeddings": [[...], [...]],
    "visits": int,
    "first_seen": "ISO timestamp",
    "last_seen": "ISO timestamp",
    "label": "unknown" or real name
  }
- find_match(emb) -> (visitor_id, score) nebo (None, 0.0)
- add_new(emb) -> visitor_id
- update_existing(visitor_id, emb) -> increments visits, appends emb (capped)
- list/delete/rename
"""

import os
import json
import time
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine


def now_iso():
    return datetime.utcnow().isoformat(sep=' ', timespec='seconds')


class VisitorDB:
    def __init__(self, path="visitors.json", threshold=0.60, max_embeddings=20):
        self.path = path
        self.threshold = threshold
        self.max_embeddings = max_embeddings
        self.db = self._load()
        # ensure sequential unknown id counter
        self._ensure_counter()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # convert embeddings lists to numpy arrays
                out = {}
                for vid, rec in data.items():
                    emb_list = [np.array(e, dtype=float) for e in rec.get("embeddings", [])]
                    out[vid] = {
                        "embeddings": emb_list,
                        "visits": rec.get("visits", 0),
                        "first_seen": rec.get("first_seen", now_iso()),
                        "last_seen": rec.get("last_seen", now_iso()),
                        "label": rec.get("label", "unknown")
                    }
                return out
            except Exception as e:
                print(f"[VisitorDB] Chyba při načítání {self.path}: {e}")
                return {}
        return {}

    def save(self):
        try:
            serial = {}
            for vid, rec in self.db.items():
                serial[vid] = {
                    "embeddings": [e.tolist() for e in rec["embeddings"]],
                    "visits": rec["visits"],
                    "first_seen": rec["first_seen"],
                    "last_seen": rec["last_seen"],
                    "label": rec["label"]
                }
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(serial, f, indent=2, ensure_ascii=False)
            #print(f"[VisitorDB] Saved {len(self.db)} visitors -> {self.path}")
        except Exception as e:
            print(f"[VisitorDB] Chyba při ukládání: {e}")

    def _ensure_counter(self):
        # find highest unknown_N index
        maxn = 0
        for vid in self.db.keys():
            if vid.startswith("unknown_"):
                try:
                    n = int(vid.split("_")[1])
                    if n > maxn: maxn = n
                except Exception:
                    pass
        self._counter = maxn

    def _next_unknown_id(self):
        self._counter += 1
        return f"unknown_{self._counter:03d}"

    def find_match(self, emb):
        """Najde nejlepší shodu v návštěvnících. Vrátí (visitor_id, score) nebo (None,0)."""
        if emb is None or len(self.db) == 0:
            return None, 0.0
        emb = np.array(emb).reshape(-1)
        best_id, best_score = None, -1.0
        for vid, rec in self.db.items():
            for e in rec["embeddings"]:
                e = np.array(e).reshape(-1)
                if e.shape != emb.shape:
                    continue
                score = 1.0 - cosine(emb, e)
                if score > best_score:
                    best_id, best_score = vid, score
        if best_score >= self.threshold:
            return best_id, float(best_score)
        return None, float(best_score)

    def add_new(self, emb):
        """Vytvoří nového visitora s jedním embeddingem."""
        emb = np.array(emb).reshape(-1)
        vid = self._next_unknown_id()
        self.db[vid] = {
            "embeddings": [emb],
            "visits": 1,
            "first_seen": now_iso(),
            "last_seen": now_iso(),
            "label": "unknown"
        }
        self.save()
        return vid

    def update_existing(self, vid, emb):
        """Přidá embedding do existujícího visitora a zvýší visits."""
        if vid not in self.db:
            return False
        emb = np.array(emb).reshape(-1)
        rec = self.db[vid]
        rec["embeddings"].append(emb)
        # cap length
        if len(rec["embeddings"]) > self.max_embeddings:
            rec["embeddings"] = rec["embeddings"][-self.max_embeddings:]
        rec["visits"] = rec.get("visits", 0) + 1
        rec["last_seen"] = now_iso()
        self.save()
        return True

    def list_visitors(self):
        return sorted([(vid, rec["label"], rec["visits"]) for vid, rec in self.db.items()])

    def delete(self, vid):
        if vid in self.db:
            del self.db[vid]
            self.save()
            return True
        return False

    def rename(self, vid, new_label, move_to_gallery=False):
        """
        Přepíše label (např. unknown_003 -> 'Alena').
        If move_to_gallery True the caller should handle copying embeddings into gallery.
        """
        if vid not in self.db:
            return False
        self.db[vid]["label"] = new_label
        self.save()
        return True

    def get_record(self, vid):
        return self.db.get(vid)
