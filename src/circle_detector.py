"""Kolečko (elipsa při šikmém pohledu) a ověření, že obrys elipsou opravdu je."""
from dataclasses import dataclass, replace
import math
import cv2
import numpy as np

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
