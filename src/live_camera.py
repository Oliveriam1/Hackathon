"""Single camera owner for independent capture, analysis and browser playback."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Protocol

import numpy as np

from .models import Point
from .streaming import LiveVideo
from .target_detector import CircleInQuadrilateralDetector, DetectionResult

logger = logging.getLogger(__name__)


class CameraLifecycle(Protocol):
    def open(self) -> None: ...
    def close(self) -> None: ...


class BufferedCamera:
    """Own capture in one worker; analysis reads only the newest available frame.

    read_frame must return/raise within a second for bounded shutdown. PiCamera's
    blocking read is usable for preview, but is not a real mission adapter.
    Source close runs in the capture worker, avoiding concurrent device access.
    """
    def __init__(self, source: CameraLifecycle, read_frame: Callable[[], np.ndarray],
                 video: LiveVideo, *, fps: float = 30) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("Capture FPS must be between 0 and 120")
        self.source, self.read_frame, self.video, self.fps = source, read_frame, video, fps
        self.captured_at = 0.0
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._cleanup_error: Exception | None = None

    def open(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Create a new BufferedCamera for each run")
        self._thread = threading.Thread(target=self._capture, name="video-capture", daemon=True)
        self._thread.start()
        try:
            # Includes normal Pi startup/exposure settling, but cannot hang INIT.
            self.video.wait_for_frame(0, timeout_s=5)
        except Exception:
            self.close()
            raise

    def _capture(self) -> None:
        try:
            self.source.open()
            next_frame_at = time.monotonic()
            while not self._stop.is_set():
                self.video.publish_frame(self.read_frame())
                next_frame_at = max(next_frame_at + 1 / self.fps, time.monotonic())
                self._stop.wait(max(0, next_frame_at - time.monotonic()))
        except Exception as exc:
            self._error = exc
            logger.exception("Video camera failed")
            self.video.set_camera_status("ERROR")
        finally:
            try:
                self.source.close()
            except Exception as exc:
                self._cleanup_error = exc
                self._error = self._error or exc
                logger.exception("Video camera cleanup failed")
            self.video.set_camera_status("ERROR" if self._error else "STOPPED")

    def read(self, timeout_s: float) -> np.ndarray:
        if self._error is not None:
            raise RuntimeError(f"Camera failed: {self._error}") from self._error
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Camera read timeout must be positive and finite")
        # Require a fresh capture after this call, not a cached frame from travel.
        sample = self.video.wait_for_frame(max(self._sequence, self.video.snapshot().sequence), timeout_s)
        self._sequence, self.captured_at = sample.sequence, sample.captured_at
        return sample.frame.copy()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                self.video.set_camera_status("ERROR")
                raise RuntimeError("Camera did not stop; source read/open must be bounded")
        if self._cleanup_error is not None:
            raise RuntimeError("Camera cleanup failed") from self._cleanup_error


class StreamingDetector:
    """Reuse the detector's full geometry without a second expensive analysis."""
    def __init__(self, detector: CircleInQuadrilateralDetector,
                 camera: BufferedCamera, video: LiveVideo) -> None:
        self.detector, self.camera, self.video = detector, camera, video

    def detect(self, frame: np.ndarray, *, largest_only: bool = False) -> DetectionResult:
        captured_at = self.camera.captured_at
        result = self.detector.detect(frame, largest_only=largest_only)
        self.video.set_detection(result, captured_at)
        return result

    def detect_circle(self, frame: np.ndarray) -> Point | None:
        result = self.detect(frame, largest_only=True)
        return result.targets[0].center if result.targets else None
