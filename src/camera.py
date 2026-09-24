"""Camera calibration loading and an explicitly opened OpenCV camera backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from .models import CameraIntrinsics, CameraMount

if TYPE_CHECKING:
    import cv2
    import numpy as np


def load_camera_calibration(path: Path) -> tuple[CameraIntrinsics, CameraMount]:
    """Load explicit intrinsics and mount from JSON; no production defaults."""
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
        intrinsics = CameraIntrinsics(**data["intrinsics"])
        mount = CameraMount(**data["mount"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid camera calibration file {path}: {exc}") from exc
    return intrinsics, mount


class OpenCVCamera:
    """Local VideoCapture backend; importing or constructing it opens no device.

    A Raspberry Pi CSI/Picamera2 backend can be added here later, separately
    from detection and geolocation. Requested resolution is checked by main.
    """

    def __init__(self, index: int = 0, intrinsics: CameraIntrinsics | None = None) -> None:
        self.index = index
        self.intrinsics = intrinsics
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> OpenCVCamera:
        import cv2

        capture = cv2.VideoCapture(self.index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera {self.index}")
            if self.intrinsics is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.intrinsics.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.intrinsics.height)
        except Exception:
            capture.release()
            raise
        self._capture = capture
        return self

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera has not been opened")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a camera frame")
        return frame

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
