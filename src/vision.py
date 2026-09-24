"""Kolečko uvnitř čtyřúhelníku, bez barevné reference a letových povelů."""
from dataclasses import dataclass
import cv2
import math
import numpy as np
import time
from .circle_detector import Circle, CircleDetector
from .models import Point
from .rectangle_detector import find_rectangles, enclosing_rectangle, find_faint_quadrilaterals
from .tracker import TargetTracker

MIN_AXIS_RATIO = 0.35  # elipsa kolečka až při ~70° od kolmého pohledu
# Sledovaný cíl: zpracovává se jen výřez kolem jeho obdélníku (rychlost ve vysokém rozlišení).
REGION_MARGIN = 0.6     # zvětšení obdélníku na každou stranu, podíl jeho velikosti
REGION_MIN_MARGIN = 40  # px, rezerva na pohyb mezi snímky
REGION_MAX_MISSED = 2   # po více snímcích bez nálezu znovu celý snímek


@dataclass(frozen=True)
class Observation:
    circles: list[Circle]       # kandidáti "kolečko v obdélníku" v tomto snímku
    status: str
    rectangles: list
    confirmed: bool             # potvrzený cíl, i během krátkého výpadku
    processing_ms: float
    target: Circle | None = None   # vyhlazená poloha potvrzeného cíle
    measured: bool = False      # False = poloha je jen predikce
    frame_size: tuple[int, int] | None = None  # (šířka, výška)
    region: tuple[int, int, int, int] | None = None  # zpracovaný výřez (x0, y0, x1, y1)

    @property
    def offset(self):
        """Odchylka cíle od středu obrazu, -1..1, kladně vpravo a dolů."""
        if self.target is None or self.frame_size is None:
            return None
        width, height = self.frame_size
        return (self.target.x - width/2) / (width/2), (self.target.y - height/2) / (height/2)

    @property
    def confirmed_center(self):
        """Střed potvrzeného cíle změřeného v tomto snímku (ne predikce), jinak None."""
        if not self.confirmed or not self.measured:
            return None
        return Point(self.target.x, self.target.y)

    @property
    def summary(self):
        return self.status.split(' | ')[0]

    def annotate(self, image):
        """Kreslení pro webový stream (streaming.draw_overlay)."""
        return annotate_observation(image, self)


class Vision:
    def __init__(self):
        self.detector = CircleDetector()
        self.tracker = TargetTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.previous_shape = None
        self.previous_sensitivity = None
        self.region_box = None  # obdélník cíle vůči středu kolečka: (x0, y0, x1, y1)
        self._edges = None

    @property
    def edges(self):
        """Hrany posledního snímku v plné velikosti (výřez vložený do černé plochy)."""
        if self._edges is None:
            return None
        edges, (x0, y0), shape = self._edges
        full = np.zeros(shape, np.uint8)
        full[y0:y0+edges.shape[0], x0:x0+edges.shape[1]] = edges
        return full

    def detect(self, frame, *, largest_only=False):
        """Rozhraní StreamingDetector (webový stream); vrací Observation."""
        return self.observe(frame)

    def detect_circle(self, frame):
        """Rozhraní mise: střed potvrzeného a právě změřeného cíle, jinak None."""
        return self.observe(frame).confirmed_center

    def observe(self, frame):
        started = time.perf_counter()
        if self.previous_shape != frame.shape or self.previous_sensitivity != self.detector.sensitivity:
            self.reset()
        self.previous_shape = frame.shape
        self.previous_sensitivity = self.detector.sensitivity
        x0, y0, x1, y1 = self._region(frame.shape)
        crop = frame[y0:y1, x0:x1]
        gray, edges = self.detector.prepare(crop)
        self._edges = (edges, (x0, y0), frame.shape[:2])
        rectangles = find_rectangles(edges)
        if not rectangles:
            rectangles = find_faint_quadrilaterals(crop)
        candidates, sources = [], []
        for rectangle in rectangles:
            for circle in self._circles_in(gray, edges, rectangle):
                if any(math.hypot(circle.x-old.x, circle.y-old.y) < max(4, old.radius*0.2)
                       for old in candidates):
                    continue
                candidates.append(circle)
                sources.append(rectangle)
        shift = np.array([x0, y0], np.int32)
        candidates = [circle.shifted(x0, y0) for circle in candidates]
        rectangles = [rectangle + shift for rectangle in rectangles]
        sources = [rectangle + shift for rectangle in sources]
        state = self.tracker.update(candidates, lambda prediction: [
            circle.shifted(x0, y0) for circle in self._circles_near(gray, edges, prediction.shifted(-x0, -y0))])
        self._remember_region(candidates, sources)
        elapsed = (time.perf_counter()-started)*1000
        height, width = frame.shape[:2]
        return Observation(candidates, f'{state.status} | {elapsed:.0f} ms', rectangles, state.confirmed,
                           elapsed, state.target, state.measured, (width, height), (x0, y0, x1, y1))

    def _region(self, shape):
        """Výřez kolem predikce sledovaného cíle, jinak celý snímek (x0, y0, x1, y1)."""
        height, width = shape[:2]
        prediction = self.tracker.predicted
        if prediction is None or self.region_box is None or self.tracker.missed > REGION_MAX_MISSED:
            return 0, 0, width, height
        bx0, by0, bx1, by1 = self.region_box
        margin = max(REGION_MIN_MARGIN, REGION_MARGIN*max(bx1-bx0, by1-by0)) * (1 + self.tracker.missed)
        x0 = int(max(0, prediction.x + bx0 - margin))
        y0 = int(max(0, prediction.y + by0 - margin))
        x1 = int(min(width, prediction.x + bx1 + margin))
        y1 = int(min(height, prediction.y + by1 + margin))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return 0, 0, width, height
        return x0, y0, x1, y1

    def _remember_region(self, candidates, sources):
        track = self.tracker.track
        if track is None:
            self.region_box = None
            return
        if self.tracker.missed or not candidates:
            return  # predikce nebo jen kolečko: obdélník zůstává z posledního nálezu
        index = min(range(len(candidates)),
                    key=lambda i: math.hypot(candidates[i].x-track.x, candidates[i].y-track.y))
        x, y, width, height = cv2.boundingRect(sources[index])
        circle = candidates[index]
        self.region_box = (x - circle.x, y - circle.y, x + width - circle.x, y + height - circle.y)

    def _circles_in(self, gray, edges, rectangle):
        x, y, width, height = cv2.boundingRect(rectangle)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [(rectangle - (x, y)).astype(np.int32)], 255)
        # Rezerva: okraj obdélníku ani kolečko, které se ho dotýká, nepatří dovnitř.
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        # Předzpracování je společné. Drahé hledání kruhů jen v oblasti terče.
        circles = self.detector.detect_prepared(gray[y:y+height, x:x+width], edges[y:y+height, x:x+width],
                                                mask, MIN_AXIS_RATIO)
        return [circle.shifted(x, y) for circle in circles
                if enclosing_rectangle(circle.shifted(x, y), [rectangle]) is not None]

    def _circles_near(self, gray, edges, prediction):
        """Samotná kolečka kolem predikce; smí jen udržet již potvrzený cíl."""
        reach = int(2.5*prediction.radius) + 10
        frame_height, frame_width = gray.shape
        x0, y0 = max(0, int(prediction.x) - reach), max(0, int(prediction.y) - reach)
        x1, y1 = min(frame_width, int(prediction.x) + reach), min(frame_height, int(prediction.y) + reach)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        circles = self.detector.detect_prepared(gray[y0:y1, x0:x1], edges[y0:y1, x0:x1],
                                                min_axis_ratio=MIN_AXIS_RATIO)
        return [circle.shifted(x0, y0) for circle in circles]


def _draw_ellipse(image, circle, color, thickness):
    center = (round(circle.x), round(circle.y))
    axes = (max(1, round(circle.radius)), max(1, round(circle.minor_radius)))
    cv2.ellipse(image, center, axes, circle.angle, 0, 360, color, thickness)
    return center


def _text(image, text, origin, color, scale):
    # Ztmavený podklad: čitelné i na světlém obrazu. (Obrys silnější čarou nesedí,
    # Hershey písmo mění šířku znaků podle tloušťky.)
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    box = image[max(0, y-height-4):y+baseline+2, max(0, x-4):x+width+4]
    box[:] = cv2.convertScaleAbs(box, alpha=0.3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def annotate_observation(image, observation):
    output = image.copy()
    height, width = output.shape[:2]
    if observation.region is not None and observation.region != (0, 0, width, height):
        x0, y0, x1, y1 = observation.region
        cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (128, 128, 128), 1)
    for rectangle in observation.rectangles:
        cv2.polylines(output, [rectangle], True, (255, 180, 0), 1)
    for circle in observation.circles:
        _draw_ellipse(output, circle, (0, 255, 255), 1)
    image_center = (width // 2, height // 2)
    cv2.drawMarker(output, image_center, (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    target = observation.target
    if target is not None:
        # Zelená = změřeno v tomto snímku, oranžová = predikce během výpadku.
        color = (0, 255, 0) if observation.measured else (0, 165, 255)
        center = _draw_ellipse(output, target, color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 12, 1)
        cv2.line(output, image_center, center, color, 1)
        dx, dy = observation.offset
        # Dole: levý horní roh patří panelu webového streamu.
        _text(output, f'{center[0]}, {center[1]} px  odchylka {dx:+.2f} {dy:+.2f}', (10, height-40), color, 0.5)
    _text(output, observation.status, (10, height-15), (255, 255, 255), 0.6)
    return output
