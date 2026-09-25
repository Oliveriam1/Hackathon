"""Detekce a sledování červené tečky v jednom snímku + kreslení náhledu."""
from dataclasses import dataclass
import time
import cv2
from .circle_detector import Circle
from .red_detector import RedDetector
from .tracker import TimedTargetTracker


@dataclass(frozen=True)
class Observation:
    circles: list[Circle]            # všichni přijatí kandidáti v tomto snímku
    status: str
    confirmed: bool                  # potvrzený cíl (i během krátkého výpadku)
    processing_ms: float
    target: Circle | None = None     # vyhlazená poloha potvrzeného cíle
    measured: bool = False           # False = poloha je jen predikce
    frame_size: tuple[int, int] | None = None  # (šířka, výška)
    measurement: Circle | None = None           # nezhlazené měření tohoto snímku
    stage_ms: dict | None = None
    red: dict | None = None          # diagnostika detektoru a trackeru


class Vision:
    def __init__(self, expected_diameter=None):
        self.detector = RedDetector(expected_diameter)
        self.tracker = TimedTargetTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.detector.reset()
        self.previous_shape = None

    def observe(self, frame, *, sample_time=None):
        started = time.perf_counter()
        if self.previous_shape != frame.shape:
            self.reset()
        self.previous_shape = frame.shape
        timestamp = time.monotonic() if sample_time is None else sample_time
        roi = self.tracker.search_roi(frame.shape, timestamp)
        candidates = self.detector.detect(frame, roi=roi)
        stages = dict(self.detector.stage_ms)
        # Oříznutý cíl nebo prázdný výřez: hned hledat v celém snímku.
        if roi is not None and (not candidates or any(r['reason'] == 'CLIPPED' for r in self.detector.reports)):
            candidates = self.detector.detect(frame)
            stages = {key: value+self.detector.stage_ms.get(key, 0) for key, value in stages.items()}
        tracking_started = time.perf_counter()
        state = self.tracker.update(candidates, sample_time=timestamp, scores=self.detector.scores)
        elapsed = (time.perf_counter()-started)*1000
        stages['tracking'] = (time.perf_counter()-tracking_started)*1000
        stages['total'] = elapsed
        height, width = frame.shape[:2]
        red = dict(geometry=self.detector.geometry, diagnostics=self.detector.diagnostics(),
                   tracking=self.tracker.diagnostics())
        return Observation(candidates, state.status, state.confirmed, elapsed, state.target, state.measured,
                           (width, height), self.tracker.last_measurement, stages, red)


def _text(image, text, origin, color, scale):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def annotate(image, observation, aim=None):
    """Kandidáti, sledovaný cíl, střed obrazu a stav míření."""
    view = image.copy()
    height, width = view.shape[:2]
    red = observation.red or {}
    for item in red.get('diagnostics', {}).get('candidates', []):
        x, y, w, h = item['box_px']
        color = (0, 220, 220) if item['accepted'] else (130, 130, 130)
        cv2.rectangle(view, (x, y), (x+w, y+h), color, 1)
        _text(view, item['reason'], (x, max(12, y-3)), color, .35)
    center = (width//2, height//2)
    cv2.drawMarker(view, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    point = observation.measurement if observation.measured else observation.target
    if point is not None:
        color = (0, 255, 0) if observation.confirmed and observation.measured else (0, 165, 255)
        position = (round(point.x), round(point.y))
        axes = (max(1, round(point.radius)), max(1, round(point.minor_radius)))
        cv2.ellipse(view, position, axes, point.angle, 0, 360, color, 2)
        cv2.drawMarker(view, position, color, cv2.MARKER_CROSS, 12, 1)
        cv2.line(view, center, position, color, 1)
        _text(view, f'odchylka {point.x-width/2:+.1f}, {point.y-height/2:+.1f} px', (10, 40), color, .5)
    if aim is not None:
        text = f'{aim.state}'
        if aim.target_angles is not None:
            text += f'  uhel R={aim.target_angles.right:+.2f}  F={aim.target_angles.forward:+.2f} deg'
        _text(view, text, (10, 20), (255, 255, 255), .55)
    _text(view, f'{observation.status} | {observation.processing_ms:.0f} ms', (10, height-15), (255, 255, 255), .5)
    return view
