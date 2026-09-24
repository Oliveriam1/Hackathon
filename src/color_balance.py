"""Optional white balance measured from a known neutral patch, not scene averages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class WhiteBalance:
    """Fixed BGR channel gains; identity by default. Input frame is never modified."""

    gains_bgr: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.gains_bgr) != 3 or any(not np.isfinite(gain) or gain <= 0 for gain in self.gains_bgr):
            raise ValueError("White-balance gains must be three finite positive numbers")

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
            raise ValueError("Expected a non-empty uint8 BGR image")

    @classmethod
    def from_reference(cls, frame: np.ndarray, roi: tuple[int, int, int, int]) -> WhiteBalance:
        """Measure a white/gray reference ROI (x,y,width,height), under scene lighting.

        The caller must choose a physically neutral patch. Do not sample the
        green marker. Avoid automatic whole-frame gray-world correction, which
        would neutralize a scene dominated by green ground or a colored marker.
        """
        cls._validate_frame(frame)
        x, y, width, height = roi
        if (x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > frame.shape[1] or y + height > frame.shape[0]):
            raise ValueError("White-balance reference must be inside the image")
        pixels = frame[y:y + height, x:x + width].reshape(-1, 3)
        usable = pixels[np.all((pixels > 15) & (pixels < 245), axis=1)]
        if len(usable) < max(16, len(pixels) // 2):
            raise ValueError("White-balance reference is too dark, clipped or too small")
        reference = np.median(usable, axis=0)
        # Attenuate excessive channels instead of amplifying noise/clipping.
        gains = reference.min() / reference
        return cls(tuple(float(gain) for gain in gains))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        self._validate_frame(frame)
        corrected = frame.astype(np.float32) * np.asarray(self.gains_bgr, dtype=np.float32)
        return np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
