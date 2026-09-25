"""Barevná segmentace a ověření všech kandidátů; identitu cíle vlastní tracker."""
import math
import time
from collections import Counter
import cv2
import numpy as np
from .circle_detector import Circle, fit_ellipse

TARGET_DIAMETER_M = 0.20
HFOV_DEG = 54.0
REDNESS_MIN, R_MIN, RED_FRACTION_MIN = 25, 50, 0.35
QUAD_MIN_AREA_PX, QUAD_MAX_FRAME_FRACTION = 300, 0.95
CIRCLE_MIN_AREA_PX, CIRCLE_MIN_CIRCULARITY, CIRCLE_MAX_ASPECT_RATIO = 12, 0.68, 1.45


def expected_diameter_px(frame_width, altitude_m, gimbal_x_deg=0., gimbal_y_deg=0.,
                         target_diameter_m=TARGET_DIAMETER_M, hfov_deg=HFOV_DEG):
    focal = frame_width / (2 * math.tan(math.radians(hfov_deg / 2)))
    return focal * target_diameter_m / altitude_m * max(0., math.cos(math.radians(gimbal_x_deg)) * math.cos(math.radians(gimbal_y_deg)))


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


def red_and_pink_circle_masks(img, *, redness_min=25, r_min=50, red_fraction_min=0.35):
    """Pouze barevné hypotézy; růžová i červená se ověřují společně později."""
    b, g, r = cv2.split(img)
    strength = cv2.subtract(r, cv2.max(g, b))
    red = (strength > redness_min) & (r > r_min) & (strength > r.astype(np.float32)*red_fraction_min)
    rg, bg = cv2.subtract(r, g), cv2.subtract(b, g)
    chroma = cv2.subtract(cv2.max(r, cv2.max(g, b)), cv2.min(r, cv2.min(g, b)))
    pink = (r > 80) & (rg > 22) & (bg > 8) & (chroma > 30)
    # Bez plošného close 3x3: neslepujeme sousední několikapixelové tečky.
    return (red | pink).astype(np.uint8)*255, cv2.max(strength, rg), pink.astype(np.uint8)*255


class RedDetector:
    def __init__(self, expected_diameter=None, altitude=None):
        self.expected_diameter = expected_diameter
        self.altitude_source = lambda: {'value_m': altitude, 'source': 'manual' if altitude else 'unknown'}
        self.gimbal_source = lambda: (0., 0.)
        self.target_diameter_m, self.hfov_deg = TARGET_DIAMETER_M, HFOV_DEG
        self.camera_model = None
        self.redness_min, self.r_min, self.red_fraction_min = 25, 50, 0.35
        self.sensitivity = 1.5
        self.analyze_geometry = False
        self.reset()

    def reset(self):
        self.edges = None
        self.geometry = None
        self.reports, self.scores = [], []
        self.stage_ms = {}
        self.scale = {}
        self.exp_px = None
        self.roi = None
        self.too_small_count = 0

    @property
    def thresholds(self):
        return dict(redness_min=self.redness_min, r_min=self.r_min, red_fraction_min=self.red_fraction_min)

    def _scale(self, width, height):
        source = self.altitude_source()
        if not isinstance(source, dict):
            source = {'value_m': source, 'source': 'manual'}
        altitude = source.get('value_m')
        expected = self.expected_diameter
        hard = expected is not None
        model = 'explicit_pixels' if hard else 'nominal_fov'
        if not hard and altitude is not None and math.isfinite(altitude) and altitude > 0:
            ax, ay = self.gimbal_source()
            expected = expected_diameter_px(width, altitude, ax, ay, self.target_diameter_m, self.hfov_deg)
            if self.camera_model is not None:
                if tuple(self.camera_model.size) != (width, height):
                    raise ValueError('Rozlišení neodpovídá kalibraci kamery.')
                focal = float(self.camera_model.matrix[0, 0])
                expected = focal*self.target_diameter_m/altitude*max(0., math.cos(math.radians(ax))*math.cos(math.radians(ay)))
                model = 'calibrated_focal'
        if expected is not None and (not math.isfinite(expected) or expected <= 0):
            expected, hard = None, False
        self.exp_px = expected
        # relative_alt není měřená vzdálenost k rovině terče. Pouze explicitní px jsou tvrdý filtr.
        self.scale = dict(expected_diameter_px=expected, hard_filter=hard,
                          altitude_m=altitude, source='explicit_pixels' if hard else source['source'], model=model)
        return expected, hard

    def detect(self, frame, roi=None):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        expected, hard = self._scale(width, height)
        x0, y0, x1, y1 = roi if roi is not None else (0, 0, width, height)
        self.roi = roi
        img = frame[y0:y1, x0:x1]
        mask, strength, pink = red_and_pink_circle_masks(img, **self.thresholds)
        masked = time.perf_counter()
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components_done = time.perf_counter()
        candidates, reports, scores = [], [], []
        self.too_small_count = int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < 3))
        eligible = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 3)
        for index in eligible[eligible != 0]:
            x, y, w, h, area = map(int, stats[index])
            major = max(w, h)
            report = dict(center_px=[float(centers[index][0]+x0), float(centers[index][1]+y0)],
                          box_px=[x+x0, y+y0, w, h], area_px=area, accepted=False, reason='TOO_SMALL')
            reports.append(report)
            if area < 3:
                continue
            if x == 0 or y == 0 or x+w == img.shape[1] or y+h == img.shape[0]:
                report['reason'] = 'CLIPPED'
                continue
            ratio = major / expected if expected else None
            if hard and not 0.4 <= ratio <= 2.5:
                report['reason'] = 'SIZE'
                continue
            component = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
            if min(w, h)/major < 0.30 or area/(w*h) < 0.40:
                report['reason'] = 'SHAPE'
                continue
            tiny = major < 10
            if tiny:
                circle = Circle(x0+float(centers[index][0]), y0+float(centers[index][1]), major/2,
                                min(w, h)/2, 0 if w >= h else 90)
            else:
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                ellipse = fit_ellipse(max(contours, key=cv2.contourArea), 0.30) if contours else None
                if ellipse is None:
                    report['reason'] = 'SHAPE'
                    continue
                circle = ellipse.shifted(x0+x, y0+y)
            # Kontrast vůči lokálnímu prstenci, nikoli celému obrazu.
            margin = max(2, min(12, major//3))
            xa, ya = max(0, x-margin), max(0, y-margin)
            xb, yb = min(img.shape[1], x+w+margin), min(img.shape[0], y+h+margin)
            ring_labels = labels[ya:yb, xa:xb]
            background = strength[ya:yb, xa:xb][ring_labels == 0]
            values = strength[y:y+h, x:x+w][component != 0]
            contrast = float(values.mean() - np.median(background)) if background.size else 0.
            report['contrast'] = contrast
            if contrast < 12:
                report['reason'] = 'LOW_CONTRAST'
                continue
            # float64 eliminuje interpretaci Nx2 float32 jako kontury v cv2.moments.
            weights = strength[y:y+h, x:x+w].astype(np.float64)*component
            moments = cv2.moments(weights)
            cx, cy = x+x0+moments['m10']/moments['m00'], y+y0+moments['m01']/moments['m00']
            circle = Circle(cx, cy, circle.radius, circle.minor_radius, circle.angle)
            quality = min(1., contrast/80)*0.25 + (0.45 if tiny else 0.65)
            if ratio is not None:
                quality -= min(0.15, abs(math.log(ratio))*0.08)
            quality = max(0., min(1., quality))
            report.update(accepted=True, reason='TINY_SHAPE_UNRESOLVED' if tiny else 'ACCEPTED',
                          score=quality, center_px=[cx, cy])
            candidates.append(circle)
            scores.append(quality)
        validated = time.perf_counter()
        self.edges = np.zeros((height, width), np.uint8)
        self.edges[y0:y1, x0:x1] = mask
        self.reports, self.scores = reports, scores
        # Volitelná geometrie pouze kolem prvního kandidáta, ne celý obraz v každém snímku.
        self.geometry = None
        if self.analyze_geometry and candidates:
            c = candidates[0]
            reach = max(50, int(c.radius*5))
            xa, ya = max(0, int(c.x)-reach), max(0, int(c.y)-reach)
            xb, yb = min(width, int(c.x)+reach), min(height, int(c.y)+reach)
            geometry = analyze_object_geometry(frame[ya:yb, xa:xb], prefer=(c.x-xa, c.y-ya))
            if geometry is not None:
                geometry['contour'] += (xa, ya)
                self.geometry = geometry  # pouze diagnostický obrys, nerozhoduje o cíli
        self.stage_ms = dict(mask=(masked-started)*1000, components=(components_done-masked)*1000,
                             validation=(validated-components_done)*1000,
                             geometry=(time.perf_counter()-validated)*1000)
        return candidates

    def diagnostics(self):
        rejections = Counter(r['reason'] for r in self.reports if not r['accepted'])
        if self.too_small_count:
            rejections['TOO_SMALL'] += self.too_small_count
        return dict(scale=self.scale, roi_px=self.roi,
                    rejections=dict(rejections),
                    # Omezujeme pouze velikost diagnostického JSON, nikoli hledání kandidátů.
                    candidates=self.reports[:100], report_count=len(self.reports),
                    reports_truncated=len(self.reports) > 100)


