"""Single camera owner for independent capture, analysis and browser playback."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Protocol

import numpy as np
import cv2

from .models import Point
from .streaming import LiveVideo
from .target_detector import CircleInQuadrilateralDetector, DetectionResult
from .tracking import EMPTY, TrackingDetector, valid_frame

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
                 video: LiveVideo, *, fps: float = 30, max_frame_gap_s: float = 5) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("Capture FPS must be between 0 and 120")
        if not math.isfinite(max_frame_gap_s) or max_frame_gap_s <= 0:
            raise ValueError("Maximum frame gap must be positive and finite")
        self.source, self.read_frame, self.video, self.fps = source, read_frame, video, fps
        self.max_frame_gap_s = max_frame_gap_s
        self.captured_at = 0.0
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._cleanup_error: Exception | None = None

    @property
    def sequence(self) -> int:
        return self._sequence

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
            last_valid_at, missing = next_frame_at, False
            while not self._stop.is_set():
                try:
                    frame = self.read_frame()
                    if not valid_frame(frame):
                        raise ValueError("Camera returned an empty or invalid frame")
                    self.video.publish_frame(frame)
                    last_valid_at = time.monotonic()
                    if missing:
                        logger.info("Camera frames recovered")
                    missing = False
                except (cv2.error, ValueError, TimeoutError) as exc:
                    if not missing:
                        logger.warning("Skipping unavailable camera frame: %s", exc)
                    missing = True
                    self.video.set_camera_status("NO FRAME")
                    if time.monotonic() - last_valid_at >= self.max_frame_gap_s:
                        raise RuntimeError("Camera supplied no valid frames within the recovery timeout") from exc
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
    """Dedicated detection/tracking worker; synchronous methods remain compatible.

    The encoder only consumes immutable geometry from LiveVideo. The worker
    drops old frames instead of queuing them and owns every OpenCV tracker.
    """
    def __init__(self, detector: CircleInQuadrilateralDetector,
                 camera: BufferedCamera, video: LiveVideo, *, detection_fps: float = 8,
                 tracking_fps: float = 30, largest_only: bool = False) -> None:
        if not math.isfinite(tracking_fps) or not 0 < tracking_fps <= 120:
            raise ValueError("Tracking FPS must be between 0 and 120")
        self.detector, self.camera, self.video = detector, camera, video
        self.tracking = TrackingDetector(detector, detection_fps=detection_fps, largest_only=largest_only)
        self.tracking_fps = tracking_fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Create a new StreamingDetector for each run")
        self._thread = threading.Thread(target=self._run, name="vision-analysis", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.error("Analysis did not stop within two seconds; native OpenCV call still running")

    def _run(self) -> None:
        sequence, last_capture = 0, 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                sample = self.video.wait_for_frame(sequence, timeout_s=0.2)
            except TimeoutError:
                self.tracking.reset()
                continue
            except RuntimeError:
                break  # Source closed or permanently failed; no busy loop.
            sequence = sample.sequence
            if sample.captured_at - last_capture > 1.0:
                self.tracking.reset()
            last_capture = sample.captured_at
            try:
                result = self.tracking.process(sample.frame, time.monotonic())
                mode = self.tracking.mode
            except Exception:
                # Includes cv2.error from tracker init/update, Hough and contours.
                logger.warning("Analysis failed; resetting and retrying", exc_info=True)
                self.tracking.reset()
                result, mode = EMPTY, "ERROR"
                self._stop.wait(1 / self.tracking.detection_fps)
            if result is not None and not self._stop.is_set():
                self.video.set_detection(result, sample.captured_at, sequence=sequence,
                                         shape=sample.frame.shape[:2], mode=mode)
            self._stop.wait(max(0, 1 / self.tracking_fps - (time.monotonic() - started)))

    def detect(self, frame: np.ndarray, *, largest_only: bool = False) -> DetectionResult:
        if self._thread is not None:
            raise RuntimeError("Synchronous detect() cannot share the analysis worker")
        captured_at = self.camera.captured_at
        result = self.detector.detect(frame, largest_only=largest_only)
        self.video.set_detection(result, captured_at)
        return result

    def detect_circle(self, frame: np.ndarray) -> Point | None:
        if self._thread is not None:
            try:
                sample = self.video.wait_for_detection(self.camera.sequence, timeout_s=0.5)
            except TimeoutError:
                return None
            if (sample.camera_status != "LIVE" or sample.detection_sequence < self.camera.sequence
                    or sample.detection_shape != frame.shape[:2]
                    or time.monotonic() - sample.detection_at > 1.0):
                return None
            result = sample.detection
            return result.targets[0].center if result is not None and result.targets else None
        result = self.detect(frame, largest_only=True)
        return result.targets[0].center if result.targets else None
