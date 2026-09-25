"""Detekce červené tečky 1:1 podle red_tracker.py (detect, růžová maska NoIR,
geometrie čtyřúhelníku a kruhů, výpis událostí, sledování ve výřezu).

Vstup je BGR (Picamera2 RGB888). Konstanty jsou stejné jako v red_tracker.py.
Rozdíl: ohnisková vzdálenost se počítá ze skutečné šířky snímku (v red_tracker.py
pevně 1296 px; při 1296x972 je výsledek totožný) a prahy červené jdou nastavit.
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
# Kamera NoIR ukáže skutečně červený povrch jako růžový/purpurový. Růžová se
# přijme jen jako přibližně kulatá souvislá oblast, ne celé růžové plochy scény.
PINK_R_MIN = 80
PINK_RG_MIN = 22
PINK_BG_MIN = 8
PINK_MIN_CHROMA = 30
PINK_MAX_ASPECT = 1.65
PINK_MIN_FILL = 0.32
PINK_MIN_CIRCULARITY = 0.45
MIN_AREA_PX = 3
SIZE_RATIO = (0.4, 2.5)   # velikost vůči očekávané, jen pro obyčejné červené skvrny
MIN_FILL = 0.3            # plocha / ohraničující obdélník (jen u skvrn >= 8 px)
HFOV_DEG = 54.0
DEFAULT_ALTITUDE_M = 20.0
ROI_HALF_MIN = 120        # poloviční velikost výřezu při sledování, px
WIDEN_AFTER_MISSES = 3    # pak hledat v celém snímku
LOST_FRAMES = 15          # pak zapomenout poslední polohu (red_tracker: zpět do SEARCH)
# --- geometrie obrysů a výpis událostí ---
ANGLE_LOG_DELTA_DEG = 5.0
EVENT_LOG_MIN_INTERVAL_S = 0.25
QUAD_MIN_AREA_PX = 300
QUAD_MAX_FRAME_FRACTION = 0.95
CIRCLE_MIN_AREA_PX = 12
CIRCLE_MIN_CIRCULARITY = 0.68
CIRCLE_MAX_ASPECT_RATIO = 1.45


def expected_diameter_px(frame_width, altitude_m, gimbal_x_deg=0.0, gimbal_y_deg=0.0):
    """Očekávaný průměr kolečka v px (red_tracker.expected_diameter_px)."""
    focal = (frame_width / 2) / math.tan(math.radians(HFOV_DEG / 2))
    slant = altitude_m / (math.cos(math.radians(gimbal_x_deg)) * math.cos(math.radians(gimbal_y_deg)))
    return focal * TARGET_DIAMETER_M / slant


# --------------------------- geometrie (red_tracker.py) ---------------------------
def normalize_rect_angle(rect):
    """Náklon ve stupních vůči nejbližší ose obrazu, [-45, +45).

    Záporně = doleva (proti směru hodin na obrazovce), kladně = doprava.
    Konvence úhlu minAreaRect se liší podle verze OpenCV; prohození šířky
    a výšky ji převede na orientaci delší strany.
    """
    (_, _), (w, h), raw_angle = rect
    if w <= 0.0 or h <= 0.0:
        return None
    angle = float(raw_angle)
    if w < h:
        angle += 90.0
    angle = (angle + 45.0) % 90.0 - 45.0
    return angle


def _contour_center(contour):
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    (x, y), _ = cv2.minEnclosingCircle(contour)
    return float(x), float(y)


def analyze_object_geometry(frame, prefer=None):
    """Hlavní čtyřúhelník a kruhové obrysy uvnitř (šedotón, Canny, geometrie obrysů).

    Vrací dict contour, rect, angle, center, circles; nebo None bez čtyřúhelníku.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3, L2gradient=False)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    max_quad_area = frame_area * QUAD_MAX_FRAME_FRACTION
    candidates = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < QUAD_MIN_AREA_PX or area > max_quad_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        contains_preferred = False
        if prefer is not None:
            contains_preferred = cv2.pointPolygonTest(
                approx, (float(prefer[0]), float(prefer[1])), False
            ) >= 0
        # Přednost má čtyřúhelník se sledovaným červeným cílem, jinak největší.
        candidates.append((1 if contains_preferred else 0, area, approx))

    if not candidates:
        return None

    _, quad_area, quad = max(candidates, key=lambda item: (item[0], item[1]))
    rect = cv2.minAreaRect(quad)
    tilt = normalize_rect_angle(rect)
    quad_center = tuple(map(float, rect[0]))

    circle_candidates = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < CIRCLE_MIN_AREA_PX or area >= quad_area * 0.35:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < CIRCLE_MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / float(min(w, h))
        if aspect > CIRCLE_MAX_ASPECT_RATIO:
            continue

        cx, cy = _contour_center(contour)
        if cv2.pointPolygonTest(quad, (float(cx), float(cy)), False) < 0:
            continue
        circle_candidates.append((area, (cx, cy), contour))

    # Vnitřní a vnější obrys téhož prstence: jeden kruh na blízká těžiště.
    circle_candidates.sort(key=lambda item: item[0], reverse=True)
    circles = []
    for area, center, contour in circle_candidates:
        _, radius = cv2.minEnclosingCircle(contour)
        duplicate = False
        for kept in circles:
            dist = math.hypot(center[0] - kept["center"][0], center[1] - kept["center"][1])
            if dist <= max(4.0, 0.5 * max(radius, kept["radius"])):
                duplicate = True
                break
        if not duplicate:
            circles.append({
                "center": center,
                "radius": float(radius),
                "area": float(area),
                "contour": contour,
            })

    return {
        "contour": quad,
        "rect": rect,
        "angle": tilt,
        "center": quad_center,
        "circles": circles,
    }


# --------------------------- barva (red_tracker.py) ---------------------------
def red_and_pink_circle_masks(img, *, redness_min=REDNESS_MIN, r_min=R_MIN, red_fraction_min=RED_FRACTION_MIN):
    """-> (maska cíle, síla červené, maska kulatých růžových oblastí)."""
    b, g, r = cv2.split(img)

    red_strength = cv2.subtract(r, cv2.max(g, b))
    _, m1 = cv2.threshold(red_strength, redness_min, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(r, r_min, 255, cv2.THRESH_BINARY)
    m3 = cv2.compare(
        red_strength,
        cv2.convertScaleAbs(r, alpha=red_fraction_min),
        cv2.CMP_GT,
    )
    red_mask = cv2.bitwise_and(cv2.bitwise_and(m1, m2), m3)

    rg = cv2.subtract(r, g)
    bg = cv2.subtract(b, g)
    maxc = cv2.max(cv2.max(r, g), b)
    minc = cv2.min(cv2.min(r, g), b)
    chroma = cv2.subtract(maxc, minc)

    _, pm1 = cv2.threshold(r, PINK_R_MIN, 255, cv2.THRESH_BINARY)
    _, pm2 = cv2.threshold(rg, PINK_RG_MIN, 255, cv2.THRESH_BINARY)
    _, pm3 = cv2.threshold(bg, PINK_BG_MIN, 255, cv2.THRESH_BINARY)
    _, pm4 = cv2.threshold(chroma, PINK_MIN_CHROMA, 255, cv2.THRESH_BINARY)
    pink_raw = cv2.bitwise_and(
        cv2.bitwise_and(pm1, pm2),
        cv2.bitwise_and(pm3, pm4),
    )
    pink_raw = cv2.morphologyEx(
        pink_raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    pink_circle_mask = np.zeros_like(pink_raw)
    contours, _ = cv2.findContours(
        pink_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        pixel_area = cv2.countNonZero(pink_raw[y:y + h, x:x + w])
        if pixel_area < MIN_AREA_PX:
            continue

        aspect = max(w, h) / float(max(1, min(w, h)))
        fill = pixel_area / float(w * h)
        if aspect > PINK_MAX_ASPECT or fill < PINK_MIN_FILL:
            continue

        if max(w, h) >= 8:
            area = abs(cv2.contourArea(contour))
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < PINK_MIN_CIRCULARITY:
                continue

        cv2.drawContours(pink_circle_mask, [contour], -1, 255, cv2.FILLED)

    target_mask = cv2.bitwise_or(red_mask, pink_circle_mask)
    target_mask = cv2.morphologyEx(
        target_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    return target_mask, red_strength, pink_circle_mask


def detect_red(frame, exp_px, roi=None, prefer=None, **thresholds):
    """red_tracker.detect: nejlepší červená/kulatá růžová skvrna -> (dict nebo None, maska)."""
    x0 = y0 = 0
    img = frame
    if roi is not None:
        x0, y0, x1, y1 = roi
        img = frame[y0:y1, x0:x1]
    mask, red_strength, pink_circle_mask = red_and_pink_circle_masks(img, **thresholds)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best, best_cost = None, float("inf")
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        major = max(bw, bh)
        ratio = major / max(float(exp_px), 1e-6)

        component_labels = (labels[by:by + bh, bx:bx + bw] == i)
        pink_here = pink_circle_mask[by:by + bh, bx:bx + bw]
        is_pink_circle = bool(np.any(component_labels & (pink_here != 0)))
        if not is_pink_circle and not (SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]):
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
    # Těžiště vážené barvou pro čistě červené i růžové/purpurové cíle.
    b, g, r = cv2.split(img)
    colour_strength = cv2.max(red_strength, cv2.subtract(r, g))
    patch = colour_strength[by:by + bh, bx:bx + bw].astype(np.float32) * (labels[by:by + bh, bx:bx + bw] == i)
    m = cv2.moments(patch)
    cx = x0 + bx + (m["m10"] / m["m00"] if m["m00"] else bw / 2)
    cy = y0 + by + (m["m01"] / m["m00"] if m["m00"] else bh / 2)
    return dict(x=cx, y=cy, w=int(bw), h=int(bh), area=int(area)), mask


class GeometryLog:
    """Výpis 'INFO: Viditelne kruhy ...' jako red_tracker.py: při změně počtu
    kruhů nebo náklonu o víc než 5°, nejvýše jednou za 0.25 s."""

    def __init__(self):
        self.count = self.angle = None
        self.last_time = 0.0

    def update(self, geometry, last_pos, frame_size, now):
        width, height = frame_size
        if geometry is not None:
            visible = len(geometry["circles"])
            angle = geometry["angle"]
            center = geometry["circles"][0]["center"] if geometry["circles"] else geometry["center"]
        else:
            visible, angle = 0, None
            center = last_pos if last_pos is not None else (width / 2.0, height / 2.0)
        count_changed = self.count is None or visible != self.count
        angle_changed = angle is not None and (self.angle is None or abs(angle - self.angle) > ANGLE_LOG_DELTA_DEG)
        if not ((count_changed or angle_changed) and now - self.last_time >= EVENT_LOG_MIN_INTERVAL_S):
            return None
        self.count = visible
        if angle is not None:
            self.angle = angle
        self.last_time = now
        angle_text = f"{angle:+.1f}" if angle is not None else "N/A"
        return (f"INFO: Viditelne kruhy: {visible}, "
                f"Stred: ({int(round(center[0]))}, {int(round(center[1]))}), Naklon: {angle_text}°")


class RedDetector:
    """Hledání a sledování jako v red_tracker.py; vrací nejvýše jeden Circle.

    expected_diameter: pevný průměr v px (jinak z výšky a úhlů závěsu).
    altitude_source / gimbal_source: funkce vracející výšku (m) a úhly závěsu (x, y) ve stupních.
    Po detect() jsou v self.geometry, self.exp_px a self.last_detection údaje pro náhled a výpis.
    """

    def __init__(self, expected_diameter=None, altitude=DEFAULT_ALTITUDE_M):
        self.expected_diameter = expected_diameter
        self.altitude_source = lambda: altitude
        self.gimbal_source = lambda: (0.0, 0.0)
        self.redness_min, self.r_min, self.red_fraction_min = REDNESS_MIN, R_MIN, RED_FRACTION_MIN
        self.analyze_geometry = True
        self.edges = None  # maska pro diagnostické zobrazení klávesou E
        self.sensitivity = 1.5
        self.reset()

    def reset(self):
        self.last_pos = None
        self.misses = 0
        self.geometry = None
        self.exp_px = None
        self.last_detection = None

    @property
    def thresholds(self):
        return dict(redness_min=self.redness_min, r_min=self.r_min, red_fraction_min=self.red_fraction_min)

    def expected_px(self, frame_width):
        if self.expected_diameter is not None:
            return self.expected_diameter
        return expected_diameter_px(frame_width, self.altitude_source(), *self.gimbal_source())

    def detect(self, frame):
        height, width = frame.shape[:2]
        exp_px = self.exp_px = self.expected_px(width)
        thresholds = self.thresholds
        roi = None
        if self.last_pos is None:
            det, mask = detect_red(frame, exp_px, **thresholds)
        else:
            half = int(max(ROI_HALF_MIN, 4 * exp_px))
            px, py = self.last_pos
            roi = (int(max(0, px - half)), int(max(0, py - half)),
                   int(min(width, px + half)), int(min(height, py + half)))
            det, mask = detect_red(frame, exp_px, roi=roi, prefer=self.last_pos, **thresholds)
            if det is None and self.misses >= WIDEN_AFTER_MISSES:
                roi = None  # rozšíření na celý snímek před ztrátou cíle
                det, mask = detect_red(frame, exp_px, prefer=self.last_pos, **thresholds)
        if roi is not None:
            full = np.zeros((height, width), np.uint8)
            full[roi[1]:roi[3], roi[0]:roi[2]] = mask
            mask = full
        self.edges = mask
        self.last_detection = det
        if det is None:
            self.misses += 1
            if self.misses > LOST_FRAMES:
                self.last_pos, self.misses = None, 0
        else:
            self.last_pos, self.misses = (det['x'], det['y']), 0
        # red_tracker.py: geometrie v každém snímku, přednost čtyřúhelníku se sledovaným cílem.
        self.geometry = (analyze_object_geometry(frame, prefer=self.last_pos if det is not None else None)
                         if self.analyze_geometry else None)
        if det is None:
            return []
        return [Circle(det['x'], det['y'], max(det['w'], det['h']) / 2)]
