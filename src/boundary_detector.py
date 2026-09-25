"""Rozhraní pro fázi detekce pásky. Zatím bez implementace a bez mapování na zem."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class BoundaryObservation:
    # Úsečky (x1, y1, x2, y2) v pixelech, počátek vlevo nahoře.
    segments_px: tuple[tuple[float, float, float, float], ...]
    sample_time: float


class BoundaryDetector(Protocol):
    def detect(self, frame: np.ndarray, *, sample_time: float) -> BoundaryObservation:
        """Najde viditelné úseky pásky; samo neurčuje bezpečný vnitřek zóny."""
        ...
