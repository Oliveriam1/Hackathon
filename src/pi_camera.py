"""Raspberry Pi CSI camera backend using Picamera2."""

from __future__ import annotations

import time

import numpy as np


class PiCamera:
    """Small Picamera2 wrapper returning NumPy frames.

    The target detector intentionally ignores colour and works from intensity only,
    so the magenta/red cast of a NoIR/infrared-sensitive camera is not relevant.
    """

    def __init__(self, index: int = 0, *, width: int = 1280, height: int = 720) -> None:
        if index < 0:
            raise ValueError("Camera index must be non-negative.")
        if width <= 0 or height <= 0:
            raise ValueError("Camera resolution must be positive.")
        self.index = index
        self.width = width
        self.height = height
        self._camera = None

    def open(self) -> None:
        if self._camera is not None:
            return

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is not available. On Raspberry Pi OS install "
                "python3-picamera2 and create the venv with --system-site-packages."
            ) from exc

        cameras = Picamera2.global_camera_info()
        if not 0 <= self.index < len(cameras):
            raise RuntimeError(
                f"CSI camera {self.index} is not available. "
                "Check with: rpicam-hello --list-cameras"
            )

        camera = Picamera2(camera_num=self.index)
        try:
            config = camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=4,
            )
            camera.configure(config)
            camera.set_controls({"AeEnable": True, "AwbEnable": True})
            camera.start()
            # Let exposure settle before the first frame.
            time.sleep(1.5)
        except BaseException:
            camera.close()
            raise

        self._camera = camera

    def read(self) -> np.ndarray:
        if self._camera is None:
            raise RuntimeError("Camera is not open.")
        frame = self._camera.capture_array("main")
        if frame is None or frame.size == 0:
            raise RuntimeError("Camera returned an empty frame.")
        return frame

    def close(self) -> None:
        if self._camera is None:
            return
        camera, self._camera = self._camera, None
        try:
            camera.stop()
        finally:
            camera.close()

    def __enter__(self) -> "PiCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
