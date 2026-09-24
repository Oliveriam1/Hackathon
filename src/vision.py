"""Pouze kruhové objekty, bez barevné reference a letových povelů."""
from dataclasses import dataclass
import cv2
import math
from .circle_detector import Circle, CircleDetector
from .rectangle_detector import find_rectangles, enclosing_rectangle


@dataclass(frozen=True)
class Observation:
    circles: list[Circle]
    status: str
    rectangles: list
    confirmed: bool


class Vision:
    def __init__(self):
        self.detector = CircleDetector()
        self.reset()

    def reset(self):
        self.previous = None
        self.hits = 0
        self.previous_shape = None
        self.previous_sensitivity = None

    def observe(self, frame):
        circles = self.detector.detect(frame)
        rectangles = find_rectangles(self.detector.edges)
        candidates = [circle for circle in circles if enclosing_rectangle(circle, rectangles) is not None]
        if self.previous_shape != frame.shape or self.previous_sensitivity != self.detector.sensitivity:
            self.reset()
        self.previous_shape = frame.shape
        self.previous_sensitivity = self.detector.sensitivity
        confirmed = False
        if len(candidates) == 1:
            current = candidates[0]
            same = (self.previous is not None
                    and math.hypot(current.x-self.previous.x, current.y-self.previous.y) < max(15, current.radius*0.75)
                    and 0.7 < current.radius/self.previous.radius < 1.4)
            self.hits = self.hits + 1 if same else 1
            self.previous = current
            confirmed = self.hits >= 4
            status = 'TERC POTVRZEN' if confirmed else f'KANDIDAT {self.hits}/4'
        else:
            self.previous = None
            self.hits = 0
            status = 'VICE KANDIDATU' if candidates else 'HLEDAM KOLECKO V OBDELNIKU'
        return Observation(candidates, f'{status} | citlivost {self.detector.sensitivity:g}', rectangles, confirmed)


def annotate_observation(image, observation):
    output = image.copy()
    for rectangle in observation.rectangles:
        cv2.polylines(output, [rectangle], True, (255, 180, 0), 1)
    color = (0, 255, 0) if observation.confirmed else (0, 255, 255)
    for circle in observation.circles:
        center = (round(circle.x), round(circle.y))
        cv2.circle(output, center, round(circle.radius), color, 2)
        cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(output, f'{center[0]}, {center[1]}',
                    (max(0, center[0]-35), max(15, center[1]-round(circle.radius)-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(output, observation.status, (10, output.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return output
