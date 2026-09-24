"""Detect a circular mark enclosed by a rectangle or general quadrilateral.

The algorithm deliberately does not use colour. It is intended for Raspberry Pi
NoIR / infrared-sensitive cameras where visible red and green hues may be badly
shifted. Geometry and intensity edges are used instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .models import Point


@dataclass(frozen=True, slots=True)
class Target:
    """Detected circle and its enclosing quadrilateral in original image pixels."""

    center: Point
    radius_px: float
    quadrilateral: tuple[Point, Point, Point, Point]
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    targets: tuple[Target, ...]
    quadrilateral_count: int


class CircleInQuadrilateralDetector:
    """Geometry-only detector for a circle inside a four-sided marker."""

    def __init__(
        self,
        *,
        min_quad_area_ratio: float = 0.01,
        max_quad_area_ratio: float = 0.95,
        min_circle_radius_px: int = 6,
    ) -> None:
        if not 0 < min_quad_area_ratio < max_quad_area_ratio <= 1:
            raise ValueError("Invalid quadrilateral area limits.")
        if min_circle_radius_px < 2:
            raise ValueError("min_circle_radius_px must be at least 2.")
        self.min_quad_area_ratio = min_quad_area_ratio
        self.max_quad_area_ratio = max_quad_area_ratio
        self.min_circle_radius_px = min_circle_radius_px

    @staticmethod
    def to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert a camera frame to intensity without relying on RGB/BGR order."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Expected a non-empty NumPy image.")
        if frame.ndim == 2:
            if frame.dtype != np.uint8:
                return np.clip(frame, 0, 255).astype(np.uint8)
            return frame.copy()
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Expected a grayscale or 3-channel image.")

        # Mean intensity is intentionally channel-order agnostic. This makes the
        # detector robust to RGB/BGR differences and to the strong colour cast of
        # infrared-sensitive cameras.
        return np.mean(frame[..., :3].astype(np.float32), axis=2).astype(np.uint8)

    @staticmethod
    def _order_quad(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        ordered = points[np.argsort(angles)]

        # Rotate so the first point is the top-left approximation.
        start = np.argmin(ordered[:, 0] + ordered[:, 1])
        ordered = np.roll(ordered, -int(start), axis=0)

        # Ensure TL, TR, BR, BL winding.
        if ordered[1, 0] < ordered[-1, 0]:
            ordered = ordered[[0, 3, 2, 1]]
        return ordered.astype(np.float32)

    def _preprocess(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.1)

        median = float(np.median(blurred))
        lower = int(max(20, 0.55 * median))
        upper = int(min(220, max(lower + 30, 1.35 * median)))
        edges = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        return enhanced, edges

    def _find_quadrilaterals(self, edges: np.ndarray) -> list[np.ndarray]:
        height, width = edges.shape
        frame_area = float(width * height)
        min_area = frame_area * self.min_quad_area_ratio
        max_area = frame_area * self.max_quad_area_ratio

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        quads: list[np.ndarray] = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 40:
                continue
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            area = abs(cv2.contourArea(polygon))
            if not min_area <= area <= max_area:
                continue

            ordered = self._order_quad(polygon[:, 0, :])
            side_lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
            if side_lengths.min() < 20:
                continue

            # Reject extremely needle-like quadrilaterals.
            if side_lengths.max() / side_lengths.min() > 8.0:
                continue

            center = ordered.mean(axis=0)
            duplicate = False
            for previous in quads:
                previous_center = previous.mean(axis=0)
                if np.linalg.norm(center - previous_center) < 10:
                    previous_area = abs(cv2.contourArea(previous))
                    if previous_area > 0 and 0.75 < area / previous_area < 1.35:
                        duplicate = True
                        break
            if not duplicate:
                quads.append(ordered)

        return sorted(quads, key=lambda q: abs(cv2.contourArea(q)), reverse=True)

    @staticmethod
    def _warp_quad(gray: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tl, tr, br, bl = quad
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)

        width = int(np.clip(round(max(top, bottom)), 80, 900))
        height = int(np.clip(round(max(left, right)), 80, 900))
        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(gray, transform, (width, height))
        return warped, transform

    def _find_circles(self, warped: np.ndarray) -> list[tuple[float, float, float, float]]:
        h, w = warped.shape
        if min(h, w) < 40:
            return []

        work = cv2.GaussianBlur(warped, (5, 5), 1.2)
        edges = cv2.Canny(work, 35, 110)
        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image_area = float(w * h)
        candidates: list[tuple[float, float, float, float]] = []

        for contour in contours:
            if len(contour) < 12:
                continue
            area = abs(cv2.contourArea(contour))
            if area < max(60.0, image_area * 0.001) or area > image_area * 0.35:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            
            # The prompt requested 0.75 to 1.2 for circularity
            if not (0.75 <= circularity <= 1.2):
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
            radius = math.sqrt(area / math.pi)

            if radius < self.min_circle_radius_px or radius > min(w, h) * 0.42:
                continue

            margin = max(4.0, radius * 0.20)
            if x - radius <= margin or y - radius <= margin or x + radius >= w - margin or y + radius >= h - margin:
                continue

            # We use circularity as our confidence score
            score = float(min(1.0, circularity) if circularity <= 1.0 else 1.0 / circularity)
            candidates.append((float(x), float(y), float(radius), score))

        return candidates

    def detect(self, frame: np.ndarray, *, largest_only: bool = False) -> DetectionResult:
        gray = self.to_gray(frame)
        _, edges = self._preprocess(gray)
        quadrilaterals = self._find_quadrilaterals(edges)

        targets: list[Target] = []
        for quad in quadrilaterals[:1] if largest_only else quadrilaterals:
            warped, transform = self._warp_quad(gray, quad)
            circles = self._find_circles(warped)
            
            for circle in circles:
                x, y, radius, confidence = circle
                inverse = np.linalg.inv(transform)
                point = np.array([[[x, y]]], dtype=np.float32)
                center_original = cv2.perspectiveTransform(point, inverse)[0, 0]

                # Radius is mapped approximately using a second point on the x axis.
                radius_point = np.array([[[x + radius, y]]], dtype=np.float32)
                radius_original = cv2.perspectiveTransform(radius_point, inverse)[0, 0]
                radius_px = float(np.linalg.norm(radius_original - center_original))

                quad_points = tuple(Point(float(px), float(py)) for px, py in quad)
                targets.append(
                    Target(
                        center=Point(float(center_original[0]), float(center_original[1])),
                        radius_px=radius_px,
                        quadrilateral=quad_points,  # type: ignore[arg-type]
                        confidence=confidence,
                    )
                )

        # De-duplicate nested/duplicate edge contours describing the same marker.
        unique: list[Target] = []
        for candidate in sorted(targets, key=lambda t: t.confidence, reverse=True):
            if any(
                math.hypot(candidate.center.x - old.center.x, candidate.center.y - old.center.y)
                < max(10.0, min(candidate.radius_px, old.radius_px) * 0.5)
                for old in unique
            ):
                continue
            unique.append(candidate)

        return DetectionResult(tuple(unique), len(quadrilaterals))

    def detect_circle(self, frame: np.ndarray) -> Point | None:
        """Return the circle center in the largest quadrilateral, or None.

        Preserve detect() for callers that want all markers. The mission uses
        this method so a smaller unrelated marker cannot replace the scan ROI.
        """
        result = self.detect(frame, largest_only=True)
        return result.targets[0].center if result.targets else None


def annotate(frame: np.ndarray, result: DetectionResult, telemetry_lines: list[str] | None = None) -> np.ndarray:
    """Draw only the detected quadrilateral/circle geometry and telemetry for testing."""
    if frame.ndim == 2:
        output = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        output = frame.copy()

    for target in result.targets:
        quad = np.array(
            [[[round(point.x), round(point.y)]] for point in target.quadrilateral],
            dtype=np.int32,
        )
        # Quad green
        cv2.polylines(output, [quad], True, (0, 255, 0), 2)
        center = (round(target.center.x), round(target.center.y))
        # Circle red
        cv2.circle(output, center, max(2, round(target.radius_px)), (0, 0, 255), 2)
        # Center red dot
        cv2.drawMarker(output, center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        
        cv2.putText(
            output,
            f"circle ({center[0]}, {center[1]}) {target.confidence:.2f}",
            (max(0, center[0] - 90), max(18, center[1] - round(target.radius_px) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    
    if telemetry_lines:
        for idx, line in enumerate(telemetry_lines):
            y = 30 + (idx * 25)
            cv2.putText(
                output,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return output
