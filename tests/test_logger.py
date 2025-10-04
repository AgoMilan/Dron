# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 11:26:29 2025

@author: Milan
"""

import sys, pathlib
import csv

# Přidáme cestu ke src, aby se našel logger.py
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from logger import CSVLogger


def test_logger_writes_and_reads(tmp_path):
    log_file = tmp_path / "test_log.csv"

    logger = CSVLogger(str(log_file))
    # frame_idx, track_id, cx, cy, w, h, state, pan_cmd, tilt_cmd
    logger.log([1, 42, 100, 200, 50, 50, "detekce", 10, 20])
    logger.log([2, 42, 105, 205, 50, 50, "predikce", 15, 25])
    logger.close()

    assert log_file.exists()

    with open(log_file, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # hlavička
    assert reader[0] == ["ts", "frame_idx", "track_id", "cx", "cy", "w", "h", "state", "pan_cmd", "tilt_cmd"]

    # první řádek: ověříme jen strukturu, ne konkrétní čas
    assert len(reader[1]) == 10
    assert reader[1][1:] == ["1", "42", "100", "200", "50", "50", "detekce", "10", "20"]

    # druhý řádek
    assert len(reader[2]) == 10
    assert reader[2][1:] == ["2", "42", "105", "205", "50", "50", "predikce", "15", "25"]
