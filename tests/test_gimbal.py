# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 12:30:28 2025

@author: Milan
"""

import sys, pathlib
import pytest

# přidáme cestu k src (stejný trik jako u loggeru)
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))

from gimbal import GimbalController


def test_gimbal_move_and_state():
    gimbal = GimbalController()

    # výchozí stav
    assert gimbal.get_state() == (0.0, 0.0)

    # pohyb
    gimbal.move_to(30.0, -10.0)
    pan, tilt = gimbal.get_state()
    assert pan == 30.0
    assert tilt == -10.0

    # log obsahuje záznam
    assert len(gimbal.log) == 1
    ts, pan_cmd, tilt_cmd = gimbal.log[0]
    assert pan_cmd == 30.0
    assert tilt_cmd == -10.0

def test_gimbal_reset():
    gimbal = GimbalController()
    gimbal.move_to(45.0, 20.0)

    # po resetu se vrátí na (0,0)
    gimbal.reset()
    pan, tilt = gimbal.get_state()
    assert (pan, tilt) == (0.0, 0.0)

    # v logu jsou dva příkazy (první pohyb a reset)
    assert len(gimbal.log) == 2
