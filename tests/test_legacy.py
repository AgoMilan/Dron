# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 09:05:49 2025

@author: Milan
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import subprocess
import sys
import pathlib
import pytest


def test_legacy_track_drone23(tmp_path):
    """
    Ověří, že původní skript track_drone23.py se spustí
    na testovacím videu a vygeneruje výstupní CSV.
    """
    project_root = pathlib.Path(__file__).resolve().parents[1]
    script_path = project_root / "src" / "legacy" / "track_drone23.py"
    video_path = project_root / "src" / "test_videos" / "drone1.mp4"

    if not video_path.exists():
        pytest.skip(f"Testovací video {video_path} neexistuje")

    out_csv = tmp_path / "out.csv"
    out_mp4 = tmp_path / "out.mp4"

    result = subprocess.run(
        [sys.executable, str(script_path),
         "--source", str(video_path),
         "--output", str(out_mp4)],
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # ověříme, že skript doběhl
    assert result.returncode == 0, "track_drone23.py crashed"

    # ověříme, že CSV bylo vytvořeno
    assert out_csv.exists(), f"Očekával jsem log {out_csv}, ale nebyl vytvořen"
