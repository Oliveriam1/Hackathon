"""KCF tracking between full circle/quad detections, owned by one analysis thread."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Callable, Protocol

import cv2
import numpy as np

from .models import Point
from .target_detector import CircleInQuadrilateralDetector, DetectionResult, Target

logger = logging.getLogger(__name__)
EMPTY = DetectionResult((), 0)


class Tracker(Protocol):
    def init(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> bool | None: ...
    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]: ...


def tracker_factory() -> Callable[[], Tracker] | None:
    """Handle both current and legacy OpenCV contrib APIs; prefer fast KCF."""
    for name in ("TrackerKCF_create", "TrackerCSRT_create"):
        for module in (cv2, getattr(cv2, "legacy", None)):
            factory = getattr(module, name, None)
            if callable(factory):
                logger.info("Tracking backend: %s", name)
                return factory
    logger.warning("KCF/CSRT unavailable: using rate-limited detection. Install OpenCV with contrib tracking.")
    return None


def valid_frame(frame: object) -> bool:
    return (isinstance(frame, np.ndarray) and frame.size > 0 and frame.dtype == np.uint8
            and (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (3, 4))))


@dataclass
class _TrackedTarget:
    original: Target
    circle: Tracker
    quad: Tracker
    circle_box: tuple[int, int, int, int]
    quad_box: tuple[int, int, int, int]


def _box(x: float, y: float, width: float, height: float,
         shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Clamp initial ROIs before passing them to OpenCV's integer Rect API."""
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(shape[1], math.ceil(x + width)), min(shape[0], math.ceil(y + height))
    if right - left < 4 or bottom - top < 4:
        raise ValueError("Tracking ROI is empty or too small")
    return left, top, right - left, bottom - top


def _valid_box(box: tuple[float, ...], shape: tuple[int, ...]) -> bool:
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return False
    x, y, width, height = box
    return x >= 0 and y >= 0 and width >= 4 and height >= 4 and x + width <= shape[1] and y + height <= shape[0]


def _map_point(point: Point, before: tuple[float, ...], after: tuple[float, ...]) -> Point:
    return Point(after[0] + (point.x - before[0]) * after[2] / before[2],
                 after[1] + (point.y - before[1]) * after[3] / before[3])


class TrackingDetector:
    """Track circle and enclosing quad independently; never run in the encoder.

    Rectangle trackers estimate translation/axis scaling, not a new homography.
    Full detection every reacquire_s corrects rotation/perspective and drift.
    Failure clears geometry immediately; reacquisition respects detection_fps.
    """
    def __init__(self, detector: CircleInQuadrilateralDetector, *, detection_fps: float = 8,
                 reacquire_s: float = 2, largest_only: bool = False,
                 factory: Callable[[], Tracker] | None = None) -> None:
        if not math.isfinite(detection_fps) or not 0 < detection_fps <= 30:
            raise ValueError("Detection FPS must be between 0 and 30")
        if not math.isfinite(reacquire_s) or reacquire_s <= 0:
            raise ValueError("Reacquisition interval must be positive and finite")
        self.detector, self.detection_fps = detector, detection_fps
        self.reacquire_s, self.largest_only = reacquire_s, largest_only
        self.factory = factory if factory is not None else tracker_factory()
        self.mode = "SEARCHING"
        self._tracks: list[_TrackedTarget] = []
        self._shape: tuple[int, ...] | None = None
        self._last_detection = -math.inf
        self._next_detection = -math.inf
        self._quad_count = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._shape = None
        self.mode = "SEARCHING"

    def process(self, frame: np.ndarray, now: float) -> DetectionResult | None:
        """None means throttled; an empty result explicitly clears old geometry."""
        if not valid_frame(frame):
            self.reset()
            return EMPTY
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape != self._shape:
            self.reset()
            self._shape = frame.shape
        if self._tracks and now - self._last_detection < self.reacquire_s:
            targets = []
            for tracked in self._tracks:
                circle_ok, circle_box = tracked.circle.update(frame)
                quad_ok, quad_box = tracked.quad.update(frame)
                if not circle_ok or not quad_ok or not _valid_box(circle_box, frame.shape) or not _valid_box(quad_box, frame.shape):
                    self.reset()
                    self.mode = "LOST"
                    return EMPTY
                center = _map_point(tracked.original.center, tracked.circle_box, circle_box)
                quad = tuple(_map_point(p, tracked.quad_box, quad_box) for p in tracked.original.quadrilateral)
                polygon = np.array([(p.x, p.y) for p in quad], np.float32)
                if cv2.pointPolygonTest(polygon, (center.x, center.y), False) < 0:
                    self.reset()
                    self.mode = "LOST"
                    return EMPTY
                scale = math.sqrt(circle_box[2] * circle_box[3] / (tracked.circle_box[2] * tracked.circle_box[3]))
                targets.append(Target(center, tracked.original.radius_px * scale, quad, tracked.original.confidence))
            self.mode = "TRACKING"
            return DetectionResult(tuple(targets), self._quad_count)
        if now < self._next_detection:
            return None
        self._next_detection = now + 1 / self.detection_fps
        self._last_detection = now
        self._tracks.clear()
        result = self.detector.detect(frame, largest_only=self.largest_only)
        self._quad_count = result.quadrilateral_count
        self.mode = "DETECTED" if result.targets else "SEARCHING"
        if self.factory is not None:
            for target in result.targets:
                radius = target.radius_px
                circle_box = _box(target.center.x - radius, target.center.y - radius,
                                  2 * radius, 2 * radius, frame.shape)
                xs, ys = [p.x for p in target.quadrilateral], [p.y for p in target.quadrilateral]
                quad_box = _box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), frame.shape)
                circle, quad = self.factory(), self.factory()
                # Modern init() returns None; legacy APIs return bool.
                if circle.init(frame, circle_box) is False or quad.init(frame, quad_box) is False:
                    self.reset()
                    return result
                self._tracks.append(_TrackedTarget(target, circle, quad, circle_box, quad_box))
        return result
