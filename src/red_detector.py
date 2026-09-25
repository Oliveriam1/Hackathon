"""Detekce červené tečky 1:1 podle red_tracker.py (funkce detect a sledování ve výřezu).

Vstup je BGR (Picamera2 RGB888). Konstanty jsou stejné jako v red_tracker.py.
Rozdíl: ohnisková vzdálenost se počítá ze skutečné šířky snímku (v red_tracker.py
pevně 1296 px; při 1296x972 je výsledek totožný).
"""
import math

import cv2
import numpy as np

from .circle_detector import Circle

# --- stejné hodnoty jako v red_tracker.py ---
TARGET_DIAMETER_M = 0.20
REDNESS_MIN = 25          # pixel je červený, když R - max(G, B) > REDNESS_MIN ...
RED_FRACTION_MIN = 0.35   # ... a ten rozdíl > RED_FRACTION_MIN * R (funguje i ve stínu) ...
R_MIN = 50                # ... a R > R_MIN
MIN_AREA_PX = 3
SIZE_RATIO = (0.4, 2.5)   # velikost skvrny vůči očekávané velikosti kolečka
MIN_FILL = 0.3            # plocha / ohraničující obdélník (jen u skvrn >= 8 px)
HFOV_DEG = 54.0
DEFAULT_ALTITUDE_M = 20.0
ROI_HALF_MIN = 120        # poloviční velikost výřezu při sledování, px
WIDEN_AFTER_MISSES = 3    # pak hledat v celém snímku
LOST_FRAMES = 15          # pak zapomenout poslední polohu (red_tracker: zpět do SEARCH)


def expected_diameter_px(frame_width, altitude_m, gimbal_x_deg=0.0, gimbal_y_deg=0.0):
    """Očekávaný průměr kolečka v px (red_tracker.expected_diameter_px)."""
    focal = (frame_width / 2) / math.tan(math.radians(HFOV_DEG / 2))
    slant = altitude_m / (math.cos(math.radians(gimbal_x_deg)) * math.cos(math.radians(gimbal_y_deg)))
    return focal * TARGET_DIAMETER_M / slant


def detect_red(frame, exp_px, roi=None, prefer=None):
    """red_tracker.detect: nejlepší červená skvrna očekávané velikosti -> (dict nebo None, maska)."""
    x0 = y0 = 0
    img = frame
    if roi is not None:
        x0, y0, x1, y1 = roi
        img = frame[y0:y1, x0:x1]
    b, g, r = cv2.split(img)
    red = cv2.subtract(r, cv2.max(g, b))
    _, m1 = cv2.threshold(red, REDNESS_MIN, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(r, R_MIN, 255, cv2.THRESH_BINARY)
    m3 = cv2.compare(red, cv2.convertScaleAbs(r, alpha=RED_FRACTION_MIN), cv2.CMP_GT)
    mask = cv2.bitwise_and(cv2.bitwise_and(m1, m2), m3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best, best_cost = None, float("inf")
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        major = max(bw, bh)
        ratio = major / exp_px
        if not (SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]):
            continue
        if major >= 8 and area / float(bw * bh) < MIN_FILL:
            continue
        cost = abs(math.log(ratio))
        if prefer is not None:
            cx, cy = x0 + bx + bw / 2, y0 + by + bh / 2
            cost += math.hypot(cx - prefer[0], cy - prefer[1]) / max(ROI_HALF_MIN, 4 * exp_px)
        if cost < best_cost:
            best_cost, best = cost, (i, bx, by, bw, bh, area)
    if best is None:
        return None, mask

    i, bx, by, bw, bh, area = best
    # Těžiště vážené mírou červené -> subpixelová přesnost u malých skvrn.
    patch = red[by:by + bh, bx:bx + bw].astype(np.float32) * (labels[by:by + bh, bx:bx + bw] == i)
    m = cv2.moments(patch)
    cx = x0 + bx + (m["m10"] / m["m00"] if m["m00"] else bw / 2)
    cy = y0 + by + (m["m01"] / m["m00"] if m["m00"] else bh / 2)
    return dict(x=cx, y=cy, w=int(bw), h=int(bh), area=int(area)), mask


class RedDetector:
    """Hledání a sledování jako v red_tracker.py; vrací nejvýše jeden Circle.

    expected_diameter: pevný průměr v px (jinak z výšky a úhlů závěsu).
    altitude_source / gimbal_source: funkce vracející výšku (m) a úhly závěsu (x, y) ve stupních.
    """

    def __init__(self, expected_diameter=None, altitude=DEFAULT_ALTITUDE_M):
        self.expected_diameter = expected_diameter
        self.altitude_source = lambda: altitude
        self.gimbal_source = lambda: (0.0, 0.0)
        self.edges = None  # maska pro diagnostické zobrazení klávesou E
        self.sensitivity = 1.5
        self.reset()

    def reset(self):
        self.last_pos = None
        self.misses = 0

    def expected_px(self, frame_width):
        if self.expected_diameter is not None:
            return self.expected_diameter
        return expected_diameter_px(frame_width, self.altitude_source(), *self.gimbal_source())

    def detect(self, frame):
        height, width = frame.shape[:2]
        exp_px = self.expected_px(width)
        roi = None
        if self.last_pos is None:
            det, mask = detect_red(frame, exp_px)
        else:
            half = int(max(ROI_HALF_MIN, 4 * exp_px))
            px, py = self.last_pos
            roi = (int(max(0, px - half)), int(max(0, py - half)),
                   int(min(width, px + half)), int(min(height, py + half)))
            det, mask = detect_red(frame, exp_px, roi=roi, prefer=self.last_pos)
            if det is None and self.misses >= WIDEN_AFTER_MISSES:
                roi = None  # rozšíření na celý snímek před ztrátou cíle
                det, mask = detect_red(frame, exp_px, prefer=self.last_pos)
        if roi is not None:
            full = np.zeros((height, width), np.uint8)
            full[roi[1]:roi[3], roi[0]:roi[2]] = mask
            mask = full
        self.edges = mask
        if det is None:
            self.misses += 1
            if self.misses > LOST_FRAMES:
                self.reset()
            return []
        self.last_pos, self.misses = (det['x'], det['y']), 0
        return [Circle(det['x'], det['y'], max(det['w'], det['h']) / 2)]
