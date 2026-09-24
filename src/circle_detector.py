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
    def __init__(self, sensitivity=2):
        self.sensitivity = sensitivity
        self.edges = None

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError('Očekáván neprázdný BGR obraz uint8.')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        threshold = {1: 100, 2: 65, 3: 40}[self.sensitivity]
        edges = cv2.Canny(gray, threshold / 2, threshold)
        self.edges = edges
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
        # Hough hledá i kruhy s přerušenými hranami. Ověříme podporu obvodu,
        # aby samotné rohy čtverců nestačily k přijetí kandidáta.
        found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=16,
                                param1=threshold, param2={1: 32, 2: 25, 3: 18}[self.sensitivity],
                                minRadius=8, maxRadius=int(min(width, height)*0.48))
        if found is not None:
            distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
            angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
            for x, y, radius in found[0]:
                if x-radius < 1 or y-radius < 1 or x+radius >= width-1 or y+radius >= height-1:
                    continue
                xs = np.rint(x+radius*np.cos(angles)).astype(int)
                ys = np.rint(y+radius*np.sin(angles)).astype(int)
                support = distance[ys, xs] <= max(2.5, radius*0.035)
                if support.mean() < {1: 0.82, 2: 0.72, 3: 0.62}[self.sensitivity]:
                    continue
                if min(section.mean() for section in np.array_split(support, 4)) < 0.4:
                    continue
                candidates.append(Circle(float(x), float(y), float(radius)))
        circles = []
        # Sloučení vnitřní a vnější hrany téhož kolečka/prstence.
        for candidate in sorted(candidates, key=lambda circle: circle.radius, reverse=True):
            if any(math.hypot(candidate.x-other.x, candidate.y-other.y) < max(4, other.radius*0.15) for other in circles):
                continue
            circles.append(candidate)
        return circles
