"""Camera-independent frame processing and the Vision program's capture loop."""

import logging
import math
import time
from collections.abc import Callable

import cv2
import numpy as np

from .camera import Camera
from .detector import ColorDetector
from .models import ColorDetection, VisionResult
from .publisher import Publisher

logger = logging.getLogger(__name__)
CandidateFilter = Callable[[np.ndarray, tuple[ColorDetection, ...]], tuple[ColorDetection, ...]]


class VisionProcessor:
    """Extract measurements, never navigation decisions or geographic coordinates.

    An optional candidate_filter can later remove tape by shape/boundary or
    temporal evidence. Surviving regions still remain unconfirmed candidates.
    """

    def __init__(self, detector: ColorDetector | None = None,
                 candidate_filter: CandidateFilter | None = None) -> None:
        self.detector = detector if detector is not None else ColorDetector()
        self.candidate_filter = candidate_filter

    def process_frame(self, frame: np.ndarray, timestamp: float | None = None) -> VisionResult:
        timestamp = time.monotonic() if timestamp is None else timestamp
        if not math.isfinite(timestamp):
            raise ValueError("Frame timestamp must be finite")
        green = self.detector.detect_green(frame)
        red = tuple(self.detector.detect_red(frame))
        if self.candidate_filter is not None:
            red = tuple(self.candidate_filter(frame, red))
        height, width = frame.shape[:2]
        return VisionResult(timestamp, width, height, green,
                            self.detector.get_centering(frame, green), red)


def draw_vision_overlay(frame: np.ndarray, result: VisionResult,
                        pink: tuple[ColorDetection, ...] | None = None) -> np.ndarray:
    """Draw candidates and centering on a copy; optional pink is debug-only."""
    overlay = frame.copy()
    groups = [
        ("GREEN", (result.green_detection,) if result.green_detection else (), (0, 255, 0)),
        ("RED CANDIDATE", result.red_candidates, (0, 0, 255)),
    ]
    if pink is not None:
        groups.append(("PINK TEST", pink, (255, 0, 255)))
    for label, detections, color in groups:
        for number, detection in enumerate(detections, start=1):
            x, y, width, height = detection.bounding_box
            center = detection.center
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
            cv2.circle(overlay, (center.x, center.y), 4, color, -1)
            text = (f"{label} {number} pixel=({center.x},{center.y}) "
                    f"area={detection.area:.0f}")
            cv2.putText(overlay, text, (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    centering = result.centering
    status = (
        f"GREEN error=({centering.error_x:+d},{centering.error_y:+d}) "
        f"{'CENTERED' if centering.centered else 'NOT CENTERED'}"
        if centering.detected else "GREEN: NOT DETECTED"
    )
    cv2.putText(overlay, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if pink is not None:
        cv2.putText(overlay, "PINK TEST: ON", (8, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 0, 255), 1, cv2.LINE_AA)
    return overlay


def run_vision(camera: Camera, processor: VisionProcessor, publisher: Publisher | None = None,
               *, preview: bool = False, test_pink: bool = False,
               max_frames: int | None = None) -> None:
    """Capture until interrupted, window closed, or max_frames reached; always close."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    window_created = False
    count = 0
    try:
        camera.start()
        while True:
            frame = camera.read()
            timestamp = time.monotonic()  # Receipt time; exposure synchronization is future work.
            result = processor.process_frame(frame, timestamp)
            if count == 0:
                logger.info("Resolution: %s x %s", result.frame_width, result.frame_height)
            logger.debug("Green detected=%s; red candidates=%s",
                         result.centering.detected, len(result.red_candidates))
            if publisher is not None:
                publisher.publish(result)
            if preview:
                # Temporary pink stays outside the GREEN/RED data contract.
                pink = tuple(processor.detector.detect_pink_test(frame)) if test_pink else None
                cv2.imshow("UAV Vision", draw_vision_overlay(frame, result, pink))
                window_created = True
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("UAV Vision", cv2.WND_PROP_VISIBLE) < 1:
                    break
            count += 1
            if max_frames is not None and count >= max_frames:
                break
    finally:
        try:
            camera.close()
        finally:
            if window_created:
                cv2.destroyAllWindows()
