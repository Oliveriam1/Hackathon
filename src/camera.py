"""Camera capture in uint8 BGR, with automatic exposure and white balance."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Explicitly opened OpenCV or Picamera2 camera; imports never open devices."""

    def __init__(self, backend: str = "opencv", index: int = 0,
                 width: int = 640, height: int = 480) -> None:
        if backend not in ("opencv", "picamera2"):
            raise ValueError("Camera backend must be opencv or picamera2")
        if index < 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid camera index or resolution")
        self.backend, self.index = backend, index
        self.width, self.height = width, height
        self._device = None

    def start(self) -> None:
        if self._device is not None:
            return
        if self.backend == "picamera2":
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise RuntimeError("Picamera2 is not available; install python3-picamera2 on the Pi") from exc
            device = Picamera2(self.index)
            try:
                # Picamera2 RGB888 means B,G,R in memory, exactly as OpenCV expects.
                # Do not apply COLOR_RGB2BGR to the resulting capture_array.
                configuration = device.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"AwbEnable": True, "AeEnable": True},
                )
                device.configure(configuration)
                device.start()
            except BaseException:
                device.close()
                raise
        else:
            import cv2

            device = cv2.VideoCapture(self.index)
            try:
                if not device.isOpened():
                    raise RuntimeError(f"Cannot open OpenCV camera {self.index}")
                device.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if not device.set(cv2.CAP_PROP_AUTO_WB, 1):
                    logger.warning("OpenCV backend cannot enable auto white balance; device settings apply")
            except BaseException:
                device.release()
                raise
        self._device = device
        try:
            # Let AWB/exposure settle before detection or a diagnostic snapshot.
            # Reading also drains queued startup frames on OpenCV backends.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                self.read()
            if self.backend == "picamera2":
                metadata = device.capture_metadata()
                logger.info("Pi camera: %s; colour gains: %s; colour temperature: %s K",
                            device.camera_properties.get("Model", "unknown"),
                            metadata.get("ColourGains"), metadata.get("ColourTemperature"))
        except BaseException:
            self.close()
            raise
        logger.info("Camera ready: %s, BGR output, automatic white balance", self.backend)

    def read(self) -> np.ndarray:
        if self._device is None:
            raise RuntimeError("Camera has not been started")
        if self.backend == "picamera2":
            frame = self._device.capture_array("main")
        else:
            success, frame = self._device.read()
            if not success:
                raise RuntimeError("Could not read a camera frame")
        import numpy as np

        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
            raise RuntimeError("Camera must return a non-empty uint8 BGR image")
        return frame

    def lock_white_balance(self) -> None:
        """Keep camera gains fixed when applying a measured software correction."""
        if self._device is None:
            raise RuntimeError("Camera has not been started")
        if self.backend == "picamera2":
            gains = self._device.capture_metadata().get("ColourGains")
            if gains is None:
                raise RuntimeError("Camera did not provide white-balance gains")
            self._device.set_controls({"AwbEnable": False, "ColourGains": gains})
        else:
            import cv2

            if not self._device.set(cv2.CAP_PROP_AUTO_WB, 0):
                raise RuntimeError("Camera cannot lock white balance for reference calibration")
        # Controls are asynchronous: discard queued frames before sampling a reference.
        for _ in range(5):
            self.read()

    def close(self) -> None:
        if self._device is not None:
            device, self._device = self._device, None
            if self.backend == "picamera2":
                try:
                    device.stop()
                finally:
                    device.close()
            else:
                device.release()
