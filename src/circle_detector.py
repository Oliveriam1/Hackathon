"""Detekce celých kruhových obrysů bez použití barvy.

Kolečko viděné šikmo je elipsa. Samostatný detektor je přísný (téměř kruh),
Vision uvnitř čtyřúhelníku povoluje elipsy parametrem min_axis_ratio.
"""
from dataclasses import dataclass, replace
import math
import cv2
import numpy as np

CANNY = {1: 100, 1.5: 82, 2: 65, 3: 40}
HOUGH_VOTES = {1: 32, 1.5: 29, 2: 25, 3: 18}
HOUGH_SUPPORT = {1: 0.82, 1.5: 0.77, 2: 0.72, 3: 0.62}
MIN_MINOR = 3.0
MIN_MAJOR = 5.0
SQUARENESS = 0.028  # 4. harmonická poloměru: kruh < 0.02, rozmazaný čtverec > 0.035


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    radius: float               # velká poloosa: při náklonu kamery se nezkracuje
    minor: float | None = None  # malá poloosa; None = kruh
    angle: float = 0.0          # směr velké osy ve stupních

    @property
    def minor_radius(self):
        return self.radius if self.minor is None else self.minor

    def shifted(self, dx, dy):
        return replace(self, x=self.x+dx, y=self.y+dy)

    def boundary(self, count=48):
        t = np.linspace(0, 2*np.pi, count, endpoint=False)
        angle = math.radians(self.angle)
        u, v = self.radius*np.cos(t), self.minor_radius*np.sin(t)
        return np.column_stack([self.x + u*math.cos(angle) - v*math.sin(angle),
                                self.y + u*math.sin(angle) + v*math.cos(angle)])


def fit_ellipse(contour, min_axis_ratio):
    """Elipsa z uzavřeného obrysu, nebo None, pokud obrys elipsou není."""
    if len(contour) < 16:
        return None
    area = cv2.contourArea(contour)
    if area < 40:
        return None
    (x, y), (width, height), angle = cv2.fitEllipse(contour)
    a, b = width/2, height/2
    major, minor = max(a, b), min(a, b)
    if minor < MIN_MINOR or major < MIN_MAJOR or minor/major < min_axis_ratio:
        return None
    # Otevřený oblouk obkreslený tam a zpět má téměř nulovou plochu.
    if abs(area/(math.pi*a*b) - 1) > 0.12 + 1.5/minor:
        return None
    points = contour[:, 0, :].astype(float) - (x, y)
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    u = points[:, 0]*cos + points[:, 1]*sin
    v = -points[:, 0]*sin + points[:, 1]*cos
    scaled = np.hypot(u/a, v/b)
    # Radiální odchylka v pixelech: čtverec má rohy o několik px mimo elipsu.
    error = np.hypot(u, v) * np.abs(1 - 1/np.maximum(scaled, 1e-6))
    if np.percentile(error, 95) > 0.7 + 0.03*major:
        return None
    phase = np.arctan2(v/b, u/a)
    sectors = np.unique((phase + np.pi) / (2*np.pi) * 24 // 1 % 24)
    if len(sectors) < 22:
        return None
    # Malý rozmazaný čtverec má po normalizaci na kruh čtyři výstupky (rohy).
    # Afinní zkreslení šikmého pohledu z něj opět udělá čtverec, test tedy platí i šikmo.
    if abs(np.mean((scaled - 1) * np.exp(-4j*phase))) > SQUARENESS:
        return None
    return Circle(float(x), float(y), major, minor, angle if a >= b else angle + 90)


def inside_mask(circle, mask):
    points = np.rint(circle.boundary()).astype(int)
    height, width = mask.shape
    if (points.min(axis=0) < 0).any() or points[:, 0].max() >= width or points[:, 1].max() >= height:
        return False
    return bool(mask[points[:, 1], points[:, 0]].all())


def merge_duplicates(candidates):
    """Sloučení vnitřní a vnější hrany téhož kolečka/prstence: zůstane největší."""
    circles = []
    for candidate in sorted(candidates, key=lambda circle: circle.radius, reverse=True):
        if any(math.hypot(candidate.x-other.x, candidate.y-other.y) < max(4, other.radius*0.2)
               for other in circles):
            continue
        circles.append(candidate)
    return circles


class CircleDetector:
    def __init__(self, sensitivity=1.5, min_axis_ratio=0.92, use_hough=False):
        self.sensitivity = sensitivity
        self.use_hough = use_hough
        self.min_axis_ratio = min_axis_ratio
        self.edges = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def prepare(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError('Očekáván neprázdný BGR obraz uint8.')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        threshold = CANNY[self.sensitivity]
        edges = cv2.Canny(gray, threshold / 2, threshold)
        self.edges = edges
        return gray, edges

    def detect(self, frame):
        gray, edges = self.prepare(frame)
        return self.detect_prepared(gray, edges)

    def detect_prepared(self, gray, edges, mask=None, min_axis_ratio=None):
        """Kolečka celá uvnitř masky; bez masky celá uvnitř výřezu s rezervou 2 px."""
        ratio = self.min_axis_ratio if min_axis_ratio is None else min_axis_ratio
        if mask is None:
            mask = np.zeros_like(gray)
            mask[2:-2, 2:-2] = 255
        candidates = []
        for source in self._binary_sources(gray, edges, mask):
            contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                circle = fit_ellipse(contour, ratio)
                if circle is not None and inside_mask(circle, mask):
                    candidates.append(circle)
        # Hough je jen záloha pro přerušené hrany. Tam, kde obrys elipsu našel,
        # by kruh opřený o okolní hrany (rám, prstenec) jen přidal chybnou polohu.
        if self.use_hough:
            candidates += [circle for circle in self._hough(gray, edges, mask)
                           if all(math.hypot(circle.x-other.x, circle.y-other.y) > circle.radius + other.radius
                                  for other in candidates)]
        return merge_duplicates(candidates)

    @staticmethod
    def _binary_sources(gray, edges, mask):
        yield cv2.bitwise_and(edges, mask)
        # Otsu dává uzavřené obrysy i tam, kde jsou hrany Canny přerušené.
        values = gray[mask > 0]
        if values.size < 50:
            return
        threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark, bright = values[values <= threshold], values[values > threshold]
        if dark.size and bright.size and bright.mean() - dark.mean() >= 25:
            # RETR_LIST vrací i díry, jedna polarita tedy stačí.
            yield np.where((gray > threshold) & (mask > 0), 255, 0).astype(np.uint8)

    def _hough(self, gray, edges, mask):
        """Téměř kolmý pohled na kruh s přerušenými hranami."""
        height, width = gray.shape
        if min(width, height) < 2*MIN_MAJOR + 4:
            return []
        found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=16,
                                 param1=CANNY[self.sensitivity], param2=HOUGH_VOTES[self.sensitivity],
                                 minRadius=8, maxRadius=int(min(width, height)*0.48))
        if found is None:
            return []
        # Ověření podpory obvodu: samotné rohy čtverců nestačí.
        distance = cv2.distanceTransform(255-cv2.bitwise_and(edges, mask), cv2.DIST_L2, 3)
        angles = np.linspace(0, 2*np.pi, 120, endpoint=False)

        def support(x, y, radius):
            xs = np.clip(np.rint(x+radius*np.cos(angles)).astype(int), 0, width-1)
            ys = np.clip(np.rint(y+radius*np.sin(angles)).astype(int), 0, height-1)
            return distance[ys, xs] <= max(2.5, radius*0.035)

        circles = []
        # HoughCircles řadí podle počtu hlasů; slabší kandidáti jsou jen šum.
        for x, y, radius in found[0][:10]:
            circle = Circle(float(x), float(y), float(radius))
            if not inside_mask(circle, mask):
                continue
            on_circle = support(x, y, radius)
            if on_circle.mean() < HOUGH_SUPPORT[self.sensitivity]:
                continue
            if min(section.mean() for section in np.array_split(on_circle, 4)) < 0.4:
                continue
            # V textuře (tráva) jsou hrany všude, i mimo obvod.
            if max(support(x, y, radius*0.8).mean(), support(x, y, radius*1.2).mean()) > on_circle.mean() - 0.35:
                continue
            circles.append(circle)
        return circles
