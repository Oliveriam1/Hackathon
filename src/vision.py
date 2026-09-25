"""Kolečko uvnitř čtyřúhelníku, bez barevné reference a letových povelů."""
from dataclasses import dataclass, replace
import cv2
import math
import numpy as np
import time
from .circle_detector import Circle, CircleDetector
from .rectangle_detector import find_rectangles, enclosing_rectangle, find_faint_quadrilaterals
from .tracker import TargetTracker, TimedTargetTracker
from .red_detector import RedDetector

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
    stage_ms: dict | None = None
    detector_mode: str = 'geometry'
    red: dict | None = None     # režim red: geometry, exp_px, detection, mask (pro náhled jako red_tracker.py)

    @property
    def offset(self):
        """Odchylka cíle od středu obrazu, -1..1, kladně vpravo a dolů."""
        if self.target is None or self.frame_size is None:
            return None
        width, height = self.frame_size
        return (self.target.x - width/2) / (width/2), (self.target.y - height/2) / (height/2)


class Vision:
    def __init__(self, mode='geometry', expected_diameter=None):
        if mode not in ('geometry', 'red'):
            raise ValueError('Neznámý režim detekce.')
        self.mode = mode
        self.detector = RedDetector(expected_diameter) if mode == 'red' else CircleDetector()
        self.tracker = TimedTargetTracker() if mode == 'red' else TargetTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        if hasattr(self.detector, 'reset'):
            self.detector.reset()  # red: zapomenout poslední polohu a výřez
        self.previous_shape = None
        self.previous_sensitivity = None

    def observe(self, frame, *, sample_time=None):
        started = time.perf_counter()
        if self.mode == 'red':
            if self.previous_shape != frame.shape:
                self.reset()
            self.previous_shape = frame.shape
            timestamp = time.monotonic() if sample_time is None else sample_time
            roi = self.tracker.search_roi(frame.shape, timestamp)
            candidates = self.detector.detect(frame, roi=roi)
            detection_stages = dict(self.detector.stage_ms)
            # Při oříznutí cíle nebo prázdném ROI rozšíříme hledání ihned.
            if roi is not None and (not candidates or any(r['reason'] == 'CLIPPED' for r in self.detector.reports)):
                candidates = self.detector.detect(frame)
                detection_stages = {key: value+self.detector.stage_ms.get(key, 0)
                                    for key, value in detection_stages.items()}
            tracking_started = time.perf_counter()
            state = self.tracker.update(candidates, sample_time=timestamp, scores=self.detector.scores)
            elapsed = (time.perf_counter()-started)*1000
            stages = detection_stages
            stages['tracking'] = (time.perf_counter()-tracking_started)*1000
            stages['total'] = elapsed
            height, width = frame.shape[:2]
            red = dict(geometry=self.detector.geometry, mask=self.detector.edges,
                       diagnostics=self.detector.diagnostics(), tracking=self.tracker.diagnostics())
            return Observation(candidates, f'{state.status} | red | {elapsed:.0f} ms', [],
                               state.confirmed, elapsed, state.target, state.measured,
                               (width, height), self.tracker.last_measurement,
                               stages, 'red', red)
        gray, edges = self.detector.prepare(frame)
        prepared = time.perf_counter()
        if self.previous_shape != frame.shape or self.previous_sensitivity != self.detector.sensitivity:
            self.reset()
        self.previous_shape = frame.shape
        self.previous_sensitivity = self.detector.sensitivity
        rectangles = find_rectangles(edges)
        if not rectangles:
            rectangles = find_faint_quadrilaterals(frame)
        quadrilaterals_done = time.perf_counter()
        candidates = []
        for rectangle in rectangles:
            for circle in self._circles_in(gray, edges, rectangle):
                if any(math.hypot(circle.x-old.x, circle.y-old.y) < max(4, old.radius*0.2)
                       for old in candidates):
                    continue
                candidates.append(circle)
        state = self.tracker.update(candidates, lambda prediction: self._circles_near(gray, edges, prediction))
        finished = time.perf_counter()
        elapsed = (finished-started)*1000
        stages = dict(prepare=(prepared-started)*1000,
                      quadrilaterals=(quadrilaterals_done-prepared)*1000,
                      circles_and_tracking=(finished-quadrilaterals_done)*1000)
        height, width = frame.shape[:2]
        return Observation(candidates, f'{state.status} | {elapsed:.0f} ms', rectangles, state.confirmed,
                           elapsed, state.target, state.measured, (width, height), self.tracker.last_measurement, stages)

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


def annotate_red(image, observation, thresholds=None):
    """Bez nové segmentace: kreslí přesné kandidáty z použitého snímku."""
    view = image.copy()
    height, width = view.shape[:2]
    red = observation.red or {}
    debug = red.get('diagnostics', {})
    for item in debug.get('candidates', []):
        x, y, w, h = item['box_px']
        color = (0, 220, 220) if item['accepted'] else (130, 130, 130)
        cv2.rectangle(view, (x, y), (x+w, y+h), color, 1)
        _text(view, item['reason'], (x, max(12, y-3)), color, .35)
    geometry = red.get('geometry')
    if geometry is not None:
        cv2.drawContours(view, [geometry['contour']], -1, (255, 0, 255), 1)
    center = (width//2, height//2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    point = observation.measurement if observation.measured else observation.target
    tracking = red.get('tracking', {})
    _text(view, f"ID: {tracking.get('track_id') or '--'}  kandidati: {len(observation.circles)}",
          (10, 20), (255, 255, 255), .5)
    if point is not None:
        color = (0, 255, 0) if observation.confirmed and observation.measured else (0, 165, 255)
        position = _draw_ellipse(view, point, color, 2)
        cv2.drawMarker(view, position, color, cv2.MARKER_CROSS, 12, 1)
        cv2.line(view, center, position, color, 1)
        kind = 'MERENI' if observation.measured else 'PREDIKCE'
        _text(view, f'{kind}: {point.x:.1f}, {point.y:.1f} px', (10, 40), color, .5)
        _text(view, f'Odchylka: {point.x-width/2:+.1f}, {point.y-height/2:+.1f} px',
              (10, 60), color, .5)
    _text(view, observation.status, (10, height-15), (255, 255, 255), .55)
    return view


def annotate_observation(image, observation):
    if observation.detector_mode == 'red' and observation.red is not None:
        return annotate_red(image, observation, observation.red.get('thresholds'))
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
