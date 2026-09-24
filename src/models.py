"""Shared image-coordinate models for UAV vision."""

import math
from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True, slots=True)
class VisionResult:
    """One frame's measurements; red regions are candidates, never confirmations.

    timestamp is local time.monotonic() in seconds at frame reception, not UTC
    or an exact sensor exposure timestamp. Compare only within the same boot.
    """

    timestamp: float
    frame_width: int
    frame_height: int
    green_detection: ColorDetection | None
    centering: CenteringResult
    red_candidates: tuple[ColorDetection, ...]


class NavigationState(Enum):
    IDLE = "IDLE"
    SEARCHING_REFERENCE = "SEARCHING_REFERENCE"
    CENTERING = "CENTERING"
    CENTERED = "CENTERED"


@dataclass(frozen=True, slots=True)
class NavigationDecision:
    """Informational state and image errors, not velocities or flight commands."""

    state: NavigationState
    horizontal_error: float
    vertical_error: float


@dataclass(frozen=True, slots=True)
class GeoCoordinate:
    """WGS84 latitude/longitude in degrees."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Invalid WGS84 latitude/longitude")


def _validate_angles(roll: float, pitch: float, yaw: float) -> None:
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("Angles must be finite degrees")


@dataclass(frozen=True, slots=True)
class Attitude:
    """FRD to NED: Rz(yaw) @ Ry(pitch) @ Rx(roll), in degrees.

    Positive roll: right wing down; pitch: nose up; yaw: North toward East.
    Yaw must reference true North, not uncorrected magnetic North.
    """

    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        _validate_angles(self.roll_deg, self.pitch_deg, self.yaw_deg)


@dataclass(frozen=True, slots=True)
class UAVTelemetry:
    """AGL is height above local ground, NOT GNSS/MSL or takeoff-relative height.

    Optional timestamp uses the same local monotonic clock as VisionResult.
    A live adapter must supply exposure-aligned, valid measurements.
    """

    position: GeoCoordinate
    altitude_agl_m: float
    attitude: Attitude
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.altitude_agl_m) or self.altitude_agl_m <= 0:
            raise ValueError("altitude_agl_m must be finite and greater than zero")
        if self.timestamp is not None and not math.isfinite(self.timestamp):
            raise ValueError("Telemetry timestamp must be finite")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole calibration in pixels; optional standard OpenCV lens coefficients."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coefficients: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for size in (self.width, self.height):
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("Image dimensions must be positive integers")
        if not all(math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Focal lengths must be positive")
        if self.distortion_coefficients is not None:
            coefficients = tuple(self.distortion_coefficients)
            if len(coefficients) not in (4, 5, 8, 12, 14):
                raise ValueError("OpenCV distortion requires 4, 5, 8, 12 or 14 coefficients")
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("Distortion coefficients must be finite")
            object.__setattr__(self, "distortion_coefficients", coefficients)

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float,
                 vertical_fov_deg: float) -> "CameraIntrinsics":
        """Approximation from explicitly supplied FOVs; never guess a camera FOV."""
        if not (0 < horizontal_fov_deg < 180 and 0 < vertical_fov_deg < 180):
            raise ValueError("FOV must be between 0 and 180 degrees")
        return cls(
            width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2)),
            height / (2 * math.tan(math.radians(vertical_fov_deg) / 2)),
            width / 2, height / 2, width, height,
        )


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Mount Euler degrees applied as Rz @ Ry @ Rx to a forward-camera basis.

    Zero mount maps camera Right/Down/Forward to body Right/Down/Forward.
    (0, -90, 0) is explicit nadir: image top faces body Forward. No default.
    """

    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        _validate_angles(self.roll_deg, self.pitch_deg, self.yaw_deg)


@dataclass(frozen=True, slots=True)
class TargetGeolocation:
    """Ground position of an input pixel; this does not confirm a target."""

    coordinate: GeoCoordinate
    north_offset_m: float
    east_offset_m: float
    ground_distance_m: float
