"""Rozhraní pro fázi detekce pásky. Zatím bez implementace a bez mapování na zem."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import cv2


@dataclass(frozen=True)
class BoundaryObservation:
    # Úsečky (x1, y1, x2, y2) v pixelech, počátek vlevo nahoře.
    segments_px: tuple[tuple[float, float, float, float], ...]
    sample_time: float


class BoundaryDetector(Protocol):
    def detect(self, frame: np.ndarray, *, sample_time: float) -> BoundaryObservation:
        """Najde viditelné úseky pásky; samo neurčuje bezpečný vnitřek zóny."""
        ...


class RedWhiteTapeDetector:
    """Barevné kandidáty pásky, nikoliv důkaz uzavřeného bezpečného obvodu.

    Omezené rozlišení a počet úseček udržují práci nezávislou na rozlišení CSI.
    Vyžaduje střídání červené/bílé podél úsečky; samotná červená tečka nestačí.
    """
    def detect(self, frame, *, sample_time):
        height, width = frame.shape[:2]
        scale = min(1., 480/max(height, width))
        small = cv2.resize(frame, (round(width*scale), round(height*scale))) if scale < 1 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red = (((h <= 12) | (h >= 170)) & (s >= 90) & (v >= 65)).astype(np.uint8)*255
        white = (s <= 65) & (v >= 140)
        lines = cv2.HoughLinesP(red, 1, np.pi/180, 18, minLineLength=45, maxLineGap=28)
        segments = []
        if lines is not None:
            # Delší úsečky mají při omezeném rozpočtu přednost.
            ordered = sorted(lines.reshape(-1, 4), key=lambda p: -(int(p[2])-int(p[0]))**2-(int(p[3])-int(p[1]))**2)
            for x0, y0, x1, y1 in ordered[:64]:
                xs = np.rint(np.linspace(x0, x1, 80)).astype(int)
                ys = np.rint(np.linspace(y0, y1, 80)).astype(int)
                labels = np.where(red[ys, xs] > 0, 1, np.where(white[ys, xs], 2, 0))
                if np.mean(labels == 1) < .2 or np.mean(labels == 2) < .2 or np.mean(labels > 0) < .85:
                    continue
                # Ignorovat jednotlivé šumové pixely; nejméně čtyři barevné běhy.
                runs = []
                start = 0
                for end in range(1, len(labels)+1):
                    if end == len(labels) or labels[end] != labels[start]:
                        if end-start >= 3 and labels[start] and (not runs or runs[-1] != labels[start]):
                            runs.append(labels[start])
                        start = end
                if len(runs) < 4:
                    continue
                candidate = tuple(float(v)/scale for v in (x0, y0, x1, y1))
                if any(np.linalg.norm(np.array(candidate[:2])-old[:2]) < 15/scale and
                       np.linalg.norm(np.array(candidate[2:])-old[2:]) < 15/scale for old in segments):
                    continue
                segments.append(candidate)
                if len(segments) >= 16:
                    break
        return BoundaryObservation(tuple(segments), sample_time)
