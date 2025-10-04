# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 12:27:52 2025

@author: Milan
"""

import time

class GimbalController:
    """
    Ovládání pan/tilt gimbalu.
    Zatím jen simulace/logování příkazů.
    Později se sem doplní skutečné ovládání motorů (PWM, serial, MAVLink).
    """

    def __init__(self):
        # aktuální poloha gimbalu
        self.pan_angle = 0.0   # horizontální osa
        self.tilt_angle = 0.0  # vertikální osa
        self.log = []          # historie příkazů

    def move_to(self, pan: float, tilt: float):
        """
        Přesune gimbal na dané úhly.
        Zatím jen uloží a zaloguje čas + cílové úhly.
        """
        timestamp = time.time()
        self.pan_angle = pan
        self.tilt_angle = tilt
        self.log.append((timestamp, pan, tilt))
        print(f"[Gimbal] {timestamp:.2f} → pan={pan:.1f}°, tilt={tilt:.1f}°")

    def get_state(self):
        """
        Vrátí aktuální polohu gimbalu (pan, tilt).
        """
        return self.pan_angle, self.tilt_angle

    def reset(self):
        """
        Vrátí gimbal do výchozí polohy (0,0).
        """
        self.move_to(0.0, 0.0)
