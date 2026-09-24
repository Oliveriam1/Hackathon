"""Červené oblasti podle dodaného red_tracker.py; vstup je BGR.

Malá barevná oblast není důkaz kruhového tvaru. Bez známé velikosti
v pixelech neodvozujeme měřítko z vymyšlené výšky dronu.
"""
import cv2
import numpy as np
from .circle_detector import Circle


class RedDetector:
    def __init__(self, expected_diameter=None):
        self.expected_diameter = expected_diameter
        self.edges = None  # pro diagnostické zobrazení masky klávesou E
        self.sensitivity = 1.5

    def detect(self, frame):
        b, g, r = cv2.split(frame)
        red = cv2.subtract(r, cv2.max(g, b))
        mask = ((red > 25) & (r > 50) &
                (red > cv2.convertScaleAbs(r, alpha=0.35))).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.edges = mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for index in range(1, count):
            x, y, width, height, area = map(int, stats[index])
            major = max(width, height)
            if area < 3:
                continue
            if self.expected_diameter is not None and not 0.4 <= major / self.expected_diameter <= 2.5:
                continue
            if major >= 8 and area / (width * height) < 0.3:
                continue
            # Vážený střed jen z pixelů této komponenty.
            # float64: OpenCV může malou float32 matici Nx2 chápat jako seznam bodů.
            patch = red[y:y+height, x:x+width].astype(np.float64)
            patch *= labels[y:y+height, x:x+width] == index
            moments = cv2.moments(patch)
            if moments['m00'] <= 0:
                continue
            candidates.append(Circle(x + moments['m10']/moments['m00'],
                                     y + moments['m01']/moments['m00'], major / 2))
        # Vracíme všechny: více červených míst nesmí automaticky znamenat zámek.
        return candidates
