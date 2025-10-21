# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 22:33:59 2025

@author: Milan
"""

# visitor_db_v2.py
# -*- coding: utf-8 -*-
"""
VisitorDB v2 – evidence návštěvníků (unknown_X)
- Automaticky ukládá nové neznámé osoby po stabilním výskytu
- Správně rozlišuje opakované návštěvy podle minimálního časového odstupu (revisit_timeout)
"""

import os
import json
import time
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


class VisitorDB:
    def __init__(self, path="visitors.json", threshold=0.60, max_embeddings=20, revisit_timeout=900):
        """
        :param path: cesta k visitors.json
        :param threshold: minimální podobnost pro shodu
        :param max_embeddings: max počet uložených embeddingů na osobu
        :param revisit_timeout: minimální odstup (v sekundách) mezi dvěma návštěvami, aby se započítala jako nová
        """
        self.path = path
        self.threshold = threshold
        self.max_embeddings = max_embeddings
        self.revisit_timeout = revisit_timeout
        self.db = self._load()
        self._ensure_counter()

    # --------------------------------------------------------
    def _load(self):
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = {}
            for vid, rec in data.items():
                out[vid] = {
                    "embeddings": [np.array(e, dtype=float) for e in rec.get("embeddings", [])],
                    "visits": rec.get("visits", 1),
                    "first_seen": rec.get("first_seen", now_iso()),
                    "last_seen": rec.get("last_seen", now_iso()),
                    "label": rec.get("label", "unknown")
                }
            return out
        except Exception as e:
            print(f"[VisitorDB] Chyba načítání {self.path}: {e}")
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
        except Exception as e:
            print(f"[VisitorDB] Chyba ukládání: {e}")

    def _ensure_counter(self):
        self._counter = 0
        for vid in self.db.keys():
            if vid.startswith("unknown_"):
                try:
                    n = int(vid.split("_")[1])
                    if n > self._counter:
                        self._counter = n
                except:
                    pass

    def _next_unknown_id(self):
        self._counter += 1
        return f"unknown_{self._counter:03d}"

    # --------------------------------------------------------
    def find_match(self, emb):
        """Najde nejlepší shodu v DB. Vrací (visitor_id, score) nebo (None,0)."""
        if emb is None or not self.db:
            return None, 0.0
        emb = np.array(emb).reshape(-1)
        best_id, best_score = None, -1.0
        for vid, rec in self.db.items():
            for e in rec["embeddings"]:
                e = np.array(e).reshape(-1)
                if e.shape != emb.shape:
                    continue
                s = 1.0 - cosine(emb, e)
                if s > best_score:
                    best_score, best_id = s, vid
        if best_score >= self.threshold:
            return best_id, float(best_score)
        return None, float(best_score)

    def add_new(self, emb):
        """Založí nového návštěvníka."""
        vid = self._next_unknown_id()
        emb = np.array(emb).reshape(-1)
        self.db[vid] = {
            "embeddings": [emb],
            "visits": 1,
            "first_seen": now_iso(),
            "last_seen": now_iso(),
            "label": "unknown"
        }
        self.save()
        print(f"[VisitorDB] Nový návštěvník uložen: {vid}")
        return vid

    def update_existing(self, vid, emb):
        """Aktualizuje záznam – navýší visits pouze pokud od poslední návštěvy uplynula revisit_timeout."""
        if vid not in self.db:
            return False
        emb = np.array(emb).reshape(-1)
        rec = self.db[vid]
        rec["embeddings"].append(emb)
        if len(rec["embeddings"]) > self.max_embeddings:
            rec["embeddings"] = rec["embeddings"][-self.max_embeddings:]

        # časové porovnání
        try:
            last_ts = time.mktime(time.strptime(rec["last_seen"], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            last_ts = 0
        now_ts = time.time()
        if now_ts - last_ts > self.revisit_timeout:
            rec["visits"] += 1

        rec["last_seen"] = now_iso()
        self.save()
        return True

    # --------------------------------------------------------
    def list_visitors(self):
        return sorted([(vid, rec["label"], rec["visits"]) for vid, rec in self.db.items()])

    def delete(self, vid):
        if vid in self.db:
            del self.db[vid]
            self.save()
            return True
        return False

    def rename(self, vid, new_label):
        if vid not in self.db:
            return False
        self.db[vid]["label"] = new_label
        self.save()
        return True

    def get_record(self, vid):
        return self.db.get(vid)
