"""HSV color detection on supplied BGR frames, without hardware access."""

import math

import cv2
import numpy as np

from .models import CenteringResult, ColorDetection, NormalizedPoint, Point


# OpenCV uses hue 0..179 and saturation/value 0..255 for uint8 images.
LOWER_GREEN = (35, 70, 60)
UPPER_GREEN = (90, 255, 255)
LOWER_RED_1 = (0, 100, 80)
UPPER_RED_1 = (10, 255, 255)
LOWER_RED_2 = (170, 100, 80)
UPPER_RED_2 = (179, 255, 255)

MIN_CONTOUR_AREA = 100.0
MORPH_KERNEL_SIZE = 5
CENTERING_TOLERANCE_PX = 10


class ColorDetector:
    """Detect the largest green reference and all sufficiently large red objects."""

    def __init__(
        self,
        *,
        min_contour_area: float = MIN_CONTOUR_AREA,
        centering_tolerance_px: int = CENTERING_TOLERANCE_PX,
    ) -> None:
        if not math.isfinite(min_contour_area) or min_contour_area < 0:
            raise ValueError("min_contour_area must be finite and non-negative")
        if not math.isfinite(centering_tolerance_px) or centering_tolerance_px < 0:
            raise ValueError("centering_tolerance_px must be finite and non-negative")
        self.min_contour_area = min_contour_area
        self.centering_tolerance_px = centering_tolerance_px
        self._kernel = np.ones(
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Require a non-empty uint8 BGR image with three channels."""
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array")
        if (
            frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            raise ValueError("frame must be a non-empty uint8 BGR image (H, W, 3)")

    def detect_green(self, frame: np.ndarray) -> ColorDetection | None:
        """Return the largest green region, or None when none passes the filter."""
        self._validate_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        detections = self._detect_regions(mask)
        return detections[0] if detections else None

    def detect_red(self, frame: np.ndarray) -> list[ColorDetection]:
        """Return red regions from both hue ranges, sorted by decreasing area."""
        self._validate_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
        mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
        return self._detect_regions(mask1 | mask2)

    def _detect_regions(self, mask: np.ndarray) -> list[ColorDetection]:
        """Clean a binary mask and extract external contour centroids."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        height, width = mask.shape
        detections: list[ColorDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_contour_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = Point(
                x=int(moments["m10"] / moments["m00"]),
                y=int(moments["m01"] / moments["m00"]),
            )
            detections.append(
                ColorDetection(
                    center=center,
                    normalized_center=NormalizedPoint(
                        x=center.x / width, y=center.y / height
                    ),
                    area=area,
                    bounding_box=cv2.boundingRect(contour),
                )
            )
        return sorted(detections, key=lambda detection: detection.area, reverse=True)

    def get_centering(
        self, frame: np.ndarray, detection: ColorDetection | None
    ) -> CenteringResult:
        """Measure offsets from (width // 2, height // 2).

        Errors are normalized by width / 2 and height / 2. A reference is
        centered when both absolute pixel errors are within the tolerance,
        including its boundary. No detection means unknown coordinates/errors.
        """
        self._validate_frame(frame)
        if detection is None:
            return CenteringResult(
                detected=False,
                center=None,
                error_x=None,
                error_y=None,
                normalized_error_x=None,
                normalized_error_y=None,
                centered=False,
            )

        height, width = frame.shape[:2]
        error_x = detection.center.x - width // 2
        error_y = detection.center.y - height // 2
        return CenteringResult(
            detected=True,
            center=detection.center,
            error_x=error_x,
            error_y=error_y,
            normalized_error_x=error_x / (width / 2),
            normalized_error_y=error_y / (height / 2),
            centered=(
                abs(error_x) <= self.centering_tolerance_px
                and abs(error_y) <= self.centering_tolerance_px
            ),
        )
