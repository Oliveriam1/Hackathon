"""Shared image-coordinate models for UAV vision."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """Pixel coordinates with origin at the top-left of the image."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class NormalizedPoint:
    """Image coordinates normalized as x / width and y / height."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class ColorDetection:
    """Contour centroid, area in pixels squared, and bounding box (x, y, w, h)."""

    center: Point
    normalized_center: NormalizedPoint
    area: float
    bounding_box: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CenteringResult:
    """Image-center offsets: positive right/down, normalized by half-size."""

    detected: bool
    center: Point | None
    error_x: int | None
    error_y: int | None
    normalized_error_x: float | None
    normalized_error_y: float | None
    centered: bool
