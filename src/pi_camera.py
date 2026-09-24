"""Raspberry Pi CSI camera backend using Picamera2.

Picamera2 is intentionally loaded only when :meth:`PiCamera.open` is called.
That keeps this module importable on macOS/Windows for demo/image testing even
though Picamera2 itself is Raspberry-Pi-specific.
"""

from __future__ import annotations

from importlib import import_module
import math
import time
from typing import Any

import numpy as np


def _load_picamera2_class() -> Any:
    """Load Picamera2 lazily and return its ``Picamera2`` class.

    Picamera2 is supplied by Raspberry Pi OS as the ``python3-picamera2``
    system package. It should not be added as a normal cross-platform pip
    dependency because it is not intended for macOS/Windows.
    """
    try:
        module = import_module("picamera2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Picamera2 is not installed or is not visible to this Python environment. "
            "On Raspberry Pi OS run: sudo apt install python3-picamera2, then create "
            "the virtual environment with: python3 -m venv --system-site-packages .venv"
        ) from exc

    picamera2_class = getattr(module, "Picamera2", None)
    if picamera2_class is None:
        raise RuntimeError(
            "The installed 'picamera2' module does not provide Picamera2. "
            "Reinstall Raspberry Pi OS package: sudo apt install --reinstall python3-picamera2"
        )
    return picamera2_class


class PiCamera:
    """Minimal Picamera2 wrapper that returns OpenCV-ready NumPy frames.

    Flow on Raspberry Pi:

        CSI camera -> Picamera2 -> NumPy ndarray -> OpenCV detector

    The detector uses image intensity/geometry rather than RED/GREEN colour, so
    the strong colour cast of a NoIR/IR-sensitive camera is not relied upon.
    """

    def __init__(self, index: int = 0, *, width: int = 1280, height: int = 720, ev: float = 0.0) -> None:
        if index < 0:
            raise ValueError("Camera index must be non-negative.")
        if width <= 0 or height <= 0:
            raise ValueError("Camera resolution must be positive.")
        if not math.isfinite(ev) or not -8 <= ev <= 8:
            raise ValueError("Exposure compensation EV must be between -8 and 8.")

        self.index = index
        self.width = width
        self.height = height
        # Kompenzace expozice: záporná EV ztmaví přesvícený bílý list.
        self.ev = ev
        self._camera: Any | None = None

    def open(self) -> None:
        """Open and start the selected Raspberry Pi CSI camera."""
        if self._camera is not None:
            return

        Picamera2 = _load_picamera2_class()

        try:
            cameras = Picamera2.global_camera_info()
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 loaded, but Raspberry Pi camera information could not be read. "
                "Check the camera connection with: rpicam-hello --list-cameras"
            ) from exc

        if not cameras:
            raise RuntimeError(
                "Picamera2 is available, but no CSI camera was detected. "
                "Check the ribbon cable and run: rpicam-hello --list-cameras"
            )
        if self.index >= len(cameras):
            raise RuntimeError(
                f"CSI camera index {self.index} is not available; detected {len(cameras)} camera(s). "
                "Check with: rpicam-hello --list-cameras"
            )

        camera = Picamera2(camera_num=self.index)
        try:
            # Picamera2 RGB888 gives BGR bytes, as OpenCV expects. The detector
            # uses channel-order-independent intensity before geometry processing.
            config = camera.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                buffer_count=4,
                controls={"AwbEnable": True, "AeEnable": True},
            )
            camera.configure(config)
            if self.ev:
                camera.set_controls({"ExposureValue": self.ev})
            camera.start()

            # Give auto-exposure/auto-white-balance a short moment to settle.
            # No colour classification is performed afterwards.
            time.sleep(1.5)
        except BaseException:
            camera.close()
            raise

        self._camera = camera

    def read(self) -> np.ndarray:
        """Capture one frame as an HxWx3 uint8 NumPy array."""
        if self._camera is None:
            raise RuntimeError("Camera is not open. Call open() first.")

        frame = self._camera.capture_array("main")
        if frame is None:
            raise RuntimeError("Picamera2 returned no frame.")

        frame = np.asarray(frame)
        if frame.size == 0:
            raise RuntimeError("Picamera2 returned an empty frame.")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"Unexpected Picamera2 frame shape: {frame.shape!r}")

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return frame

    def close(self) -> None:
        """Stop and release the CSI camera."""
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
