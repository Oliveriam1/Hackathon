"""Detekce zelené referenční tečky (místo startu) ve snímku.

Tráva je také zelená, proto nestačí „největší zelená plocha“. Kandidát musí:
- mít sytou, čistou zelenou (HSV práh, laditelný),
- mít velikost odpovídající průměru tečky při aktuální výšce (±50 % / +100 %),
- být kulatý (kruhovitost >= 0.6),
- být izolovaný: v prstenci kolem něj smí být jen málo pixelů téže barvy.
Výběr mezi kandidáty dělá volající podle očekávané polohy na zemi.
"""
import math
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerCandidate:
    x: float
    y: float
    diameter_px: float
    circularity: float
    isolation: float   # podíl prstence, který NENÍ stejné barvy (1 = dokonale izolovaný)


class GreenMarkerDetector:
    def __init__(self, *, hue=(45, 90), s_min=120, v_min=80, circularity_min=.6,
                 size_range=(.5, 2.), isolation_min=.7, max_side=640):
        self.hue, self.s_min, self.v_min = hue, s_min, v_min
        self.circularity_min, self.size_range = circularity_min, size_range
        self.isolation_min, self.max_side = isolation_min, max_side
        self.kernel = np.ones((3, 3), np.uint8)

    def mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array((self.hue[0], self.s_min, self.v_min), np.uint8)
        upper = np.array((self.hue[1], 255, 255), np.uint8)
        return cv2.morphologyEx(cv2.inRange(hsv, lower, upper), cv2.MORPH_OPEN, self.kernel)

    def detect(self, frame, expected_diameter_px=None):
        height, width = frame.shape[:2]
        scale = min(1., self.max_side/max(height, width))
        small = cv2.resize(frame, (round(width*scale), round(height*scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        mask = self.mask(small)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 6:
                continue
            diameter = 2*math.sqrt(area/math.pi)/scale
            if expected_diameter_px is not None:
                ratio = diameter/expected_diameter_px
                if not self.size_range[0] <= ratio <= self.size_range[1]:
                    continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4*math.pi*area/(perimeter*perimeter) if perimeter else 0.
            if circularity < self.circularity_min:
                continue
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [contour], -1, 255, -1)
            grow = max(3, int(diameter*scale*.5))
            ring = cv2.dilate(blob, np.ones((grow, grow), np.uint8)) & ~cv2.dilate(blob, self.kernel)
            ring_pixels = np.count_nonzero(ring)
            isolation = 1-np.count_nonzero(ring & mask)/ring_pixels if ring_pixels else 0.
            if isolation < self.isolation_min:
                continue
            m = cv2.moments(contour)
            result.append(MarkerCandidate(m['m10']/m['m00']/scale, m['m01']/m['m00']/scale,
                                          diameter, circularity, isolation))
        return result
