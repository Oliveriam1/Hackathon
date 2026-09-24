"""Pozorování reference a cíle. Nevydává letové povely ani GNSS souřadnice."""

from dataclasses import dataclass
import cv2
from .detector import ColorDetector
from .models import ColorDetection, CenteringResult


@dataclass(frozen=True)
class Observation:
    green: ColorDetection | None
    reds: list[ColorDetection]
    centering: CenteringResult
    reference_seen: bool
    status: str


class Vision:
    def __init__(self):
        self.detector = ColorDetector()
        self.reference_seen = False

    def reset(self):
        self.reference_seen = False

    def observe(self, frame) -> Observation:
        green = self.detector.detect_green(frame)
        reds = self.detector.detect_red(frame)
        self.reference_seen = self.reference_seen or green is not None
        # Více červených ploch není jednoznačný cíl; žádnou automaticky nevybíráme.
        target = reds[0] if self.reference_seen and len(reds) == 1 else None
        centering = self.detector.get_centering(frame, target)
        if not self.reference_seen:
            status = 'HLEDAM ZELENOU REFERENCI'
        elif len(reds) > 1:
            status = 'VICE CERVENYCH OBLASTI - NEJEDNOZNACNY CIL'
        elif not reds:
            status = 'REFERENCE VIDENA - HLEDAM CERVENY CIL'
        elif centering.centered:
            status = 'CERVENY BOD U STREDU OBRAZU'
        else:
            status = f'CIL: dx={centering.error_x} dy={centering.error_y} px'
        return Observation(green, reds, centering, self.reference_seen, status)


def annotate_observation(image, observation: Observation):
    """Vrací kopii náhledu s detekcemi; ASCII popisky kvůli fontu OpenCV."""
    output = image.copy()
    detections = [(red, (0, 0, 255)) for red in observation.reds]
    if observation.green is not None:
        detections.append((observation.green, (0, 255, 0)))
    for detection, color in detections:
        x, y, width, height = detection.bounding_box
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        cv2.drawMarker(output, (detection.center.x, detection.center.y), color,
                       cv2.MARKER_CROSS, 20, 2)
    height = output.shape[0]
    cv2.putText(output, observation.status, (10, height - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(output, 'POUZE OBRAZ: zona / GNSS / rizeni letu nezapojeno', (10, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return output
