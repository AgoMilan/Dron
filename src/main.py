# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:48:37 2025

@author: Milan
"""

import argparse
from tracker import DroneTrackerApp


def parse_args():
    parser = argparse.ArgumentParser(description="Sledování dronu – modulární verze")

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Zdroj videa (soubor nebo číslo kamery, např. 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/out.mp4",
        help="Výstupní video"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Zobrazovat okno s výsledkem"
    )
    parser.add_argument(
        "--pause_frame",
        type=int,
        default=15,
        help="Frame, na kterém se zastaví video pro výběr ROI"
    )
    parser.add_argument(
        "--winsize",
        type=int,
        default=640,
        help="Velikost okna pro zobrazení videa (šířka)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="KCF",
        choices=["KCF", "CSRT", "MOSSE"],
        help="Typ trackeru, který se má použít"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = DroneTrackerApp(args)
    app.run()
