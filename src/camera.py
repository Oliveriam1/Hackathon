"""BGR camera backends; devices and optional libraries are opened only by start()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class Camera(Protocol):
    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


def _validate_size(width: int, height: int) -> None:
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        raise ValueError("Camera width and height must be positive integers")


class OpenCVCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        _validate_size(width, height)
        self.index, self.width, self.height = index, width, height
        self._capture = None

    def start(self) -> None:
        import cv2

        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self.index)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open OpenCV camera {self.index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except BaseException:
            capture.release()
            raise
        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera has not been started")
        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Could not read a camera frame")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class Picamera2Camera:
    """Pi CSI backend; RGB888 produces BGR byte order for OpenCV.

    Picamera2 uses libcamera format names: RGB888 arrays are [B, G, R].
    No RGB/BGR swap is needed. See Raspberry Pi's Picamera2 manual section 4.2.2.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        _validate_size(width, height)
        self.index, self.width, self.height = index, width, height
        self._camera = None

    def start(self) -> None:
        if self._camera is not None:
            return
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError("Picamera2 is not available; install python3-picamera2 on Raspberry Pi") from exc
        camera = Picamera2(self.index)
        try:
            configuration = camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            camera.configure(configuration)
            camera.start()
        except BaseException:
            camera.close()
            raise
        self._camera = camera

    def read(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("Camera has not been started")
        frame = self._camera.capture_array("main")
        if frame is None:
            raise RuntimeError("Could not read a Picamera2 frame")
        return frame

    def close(self) -> None:
        if self._camera is not None:
            camera, self._camera = self._camera, None
            try:
                camera.stop()
            finally:
                camera.close()


def create_camera(backend: str, index: int, width: int, height: int) -> Camera:
    if backend == "opencv":
        return OpenCVCamera(index, width, height)
    if backend == "picamera2":
        return Picamera2Camera(index, width, height)
    raise ValueError(f"Unknown camera backend: {backend}")


def run_camera_preview(camera_index: int = 0, *, test_pink: bool = False) -> None:
    """Compatibility shim for the old main.py; processing lives in vision.py."""
    from .vision import VisionProcessor, run_vision

    run_vision(OpenCVCamera(camera_index), VisionProcessor(), preview=True, test_pink=test_pink)
