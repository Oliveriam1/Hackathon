"""Přísná detekce celých kruhových obrysů bez použití barvy."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float


class CircleDetector:
    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError('Očekáván neprázdný BGR obraz uint8.')
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 1)
        edges = cv2.Canny(gray, 40, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        height, width = gray.shape
        candidates = []
        for contour in contours:
            if len(contour) < 24:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 150 or perimeter == 0 or 4 * math.pi * area / perimeter**2 < 0.86:
                continue
            if len(cv2.approxPolyDP(contour, 0.02 * perimeter, True)) < 8:
                continue
            (x, y), (axis1, axis2), _ = cv2.fitEllipse(contour)
            if min(axis1, axis2) / max(axis1, axis2) < 0.92:
                continue
            distances = np.linalg.norm(contour[:, 0, :].astype(float) - [x, y], axis=1)
            radius = float(np.median(distances))
            if radius < 8 or radius > min(width, height) * 0.45:
                continue
            if x-radius <= 1 or y-radius <= 1 or x+radius >= width-2 or y+radius >= height-2:
                continue
            if np.percentile(np.abs(distances-radius), 95) / radius > 0.06:
                continue
            candidates.append(Circle(float(x), float(y), radius))
        circles = []
        # Sloučení vnitřní a vnější hrany téhož kolečka/prstence.
        for candidate in sorted(candidates, key=lambda circle: circle.radius, reverse=True):
            if any(math.hypot(candidate.x-other.x, candidate.y-other.y) < max(4, other.radius*0.15) for other in circles):
                continue
            circles.append(candidate)
        return circles
