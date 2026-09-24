"""Pouze kruhové objekty, bez barevné reference a letových povelů."""
from dataclasses import dataclass
import cv2
from .circle_detector import Circle, CircleDetector


@dataclass(frozen=True)
class Observation:
    circles: list[Circle]
    status: str


class Vision:
    def __init__(self):
        self.detector = CircleDetector()

    def observe(self, frame):
        circles = self.detector.detect(frame)
        return Observation(circles, f'KOLECKA: {len(circles)} | citlivost {self.detector.sensitivity} (1-3) | E: hrany')


def annotate_observation(image, observation):
    output = image.copy()
    for circle in observation.circles:
        center = (round(circle.x), round(circle.y))
        cv2.circle(output, center, round(circle.radius), (0, 255, 255), 2)
        cv2.drawMarker(output, center, (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(output, f'{center[0]}, {center[1]}',
                    (max(0, center[0]-35), max(15, center[1]-round(circle.radius)-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(output, observation.status, (10, output.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return output
