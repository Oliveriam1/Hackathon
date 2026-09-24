"""Kolečko uvnitř čtyřúhelníku, bez barevné reference a letových povelů."""
from dataclasses import dataclass, replace
import cv2
import math
import numpy as np
import time
from .circle_detector import Circle, CircleDetector
from .rectangle_detector import find_rectangles, enclosing_rectangle, find_faint_quadrilaterals
from .tracker import TargetTracker

MIN_AXIS_RATIO = 0.35  # elipsa kolečka až při ~70° od kolmého pohledu


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
    measurement: Circle | None = None

    @property
    def offset(self):
        """Odchylka cíle od středu obrazu, -1..1, kladně vpravo a dolů."""
        if self.target is None or self.frame_size is None:
            return None
        width, height = self.frame_size
        return (self.target.x - width/2) / (width/2), (self.target.y - height/2) / (height/2)


class Vision:
    def __init__(self):
        self.detector = CircleDetector()
        self.tracker = TargetTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.previous_shape = None
        self.previous_sensitivity = None

    def observe(self, frame):
        started = time.perf_counter()
        gray, edges = self.detector.prepare(frame)
        if self.previous_shape != frame.shape or self.previous_sensitivity != self.detector.sensitivity:
            self.reset()
        self.previous_shape = frame.shape
        self.previous_sensitivity = self.detector.sensitivity
        rectangles = find_rectangles(edges)
        if not rectangles:
            rectangles = find_faint_quadrilaterals(frame)
        candidates = []
        for rectangle in rectangles:
            for circle in self._circles_in(gray, edges, rectangle):
                if any(math.hypot(circle.x-old.x, circle.y-old.y) < max(4, old.radius*0.2)
                       for old in candidates):
                    continue
                candidates.append(circle)
        state = self.tracker.update(candidates, lambda prediction: self._circles_near(gray, edges, prediction))
        elapsed = (time.perf_counter()-started)*1000
        height, width = frame.shape[:2]
        return Observation(candidates, f'{state.status} | {elapsed:.0f} ms', rectangles, state.confirmed,
                           elapsed, state.target, state.measured, (width, height), self.tracker.last_measurement)

    def _circles_in(self, gray, edges, rectangle):
        x, y, width, height = cv2.boundingRect(rectangle)
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [(rectangle - (x, y)).astype(np.int32)], 255)
        # Rezerva: okraj obdélníku ani kolečko, které se ho dotýká, nepatří dovnitř.
        small = min(width, height) < 80
        mask = cv2.erode(mask, np.ones((3, 3) if small else (5, 5), np.uint8))
        # Předzpracování je společné. Drahé hledání kruhů jen v oblasti terče.
        local_gray = gray[y:y+height, x:x+width]
        local_edges = edges[y:y+height, x:x+width]
        if small:
            local_gray = cv2.resize(local_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            local_edges = cv2.Canny(local_gray, 25, 65)
            mask = cv2.resize(mask, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        circles = self.detector.detect_prepared(local_gray, local_edges, mask, MIN_AXIS_RATIO)
        if small:
            circles = [replace(c, x=(c.x+0.5)/2-0.5, y=(c.y+0.5)/2-0.5,
                               radius=c.radius/2, minor=c.minor_radius/2) for c in circles]
        return [circle.shifted(x, y) for circle in circles
                if enclosing_rectangle(circle.shifted(x, y), [rectangle], margin=1 if small else 2) is not None]

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
    # Tmavý obrys: čitelné i na světlém podkladu.
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def annotate_observation(image, observation):
    output = image.copy()
    height, width = output.shape[:2]
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
        _text(output, f'{center[0]}, {center[1]} px  odchylka {dx:+.2f} {dy:+.2f}', (10, 20), color, 0.5)
    _text(output, observation.status, (10, height-15), (255, 255, 255), 0.6)
    return output
