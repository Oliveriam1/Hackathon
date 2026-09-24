"""Shared image, calibration and geographic models for UAV vision."""

import math
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


@dataclass(frozen=True, slots=True)
class FieldPoint:
    """Normalized field coordinates in [0, 1], increasing right and down."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("Field coordinates must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class GridCell:
    """Zero-based row/column and a label such as A1 or G4."""

    row: int
    column: int
    label: str


@dataclass(frozen=True, slots=True)
class FieldCorners:
    """Image-space corners in perimeter order: TL, TR, BR, BL."""

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point


@dataclass(frozen=True, slots=True)
class GeoCoordinate:
    """WGS84 geodetic latitude/longitude in degrees."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Invalid WGS84 latitude or longitude")


def _validate_angles(roll: float, pitch: float, yaw: float) -> None:
    if not all(math.isfinite(angle) for angle in (roll, pitch, yaw)):
        raise ValueError("Orientation angles must be finite degrees")


@dataclass(frozen=True, slots=True)
class Attitude:
    """FRD body to NED Euler angles: Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Degrees; positive roll lowers the right wing, positive pitch raises the
    nose, positive yaw turns North toward East. Yaw is relative to true North.
    """

    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        _validate_angles(self.roll_deg, self.pitch_deg, self.yaw_deg)


@dataclass(frozen=True, slots=True)
class UAVTelemetry:
    """Telemetry at frame exposure time; AGL is height above the ground plane.

    GNSS ellipsoidal/MSL height and altitude relative to takeoff are NOT AGL.
    This version assumes camera and GNSS/body origins are colocated.
    """

    position: GeoCoordinate
    altitude_agl_m: float
    attitude: Attitude

    def __post_init__(self) -> None:
        if not math.isfinite(self.altitude_agl_m) or self.altitude_agl_m <= 0:
            raise ValueError("altitude_agl_m must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole calibration in pixels for the specified, unrotated image size.

    Optional OpenCV distortion order: k1,k2,p1,p2[,k3[,k4,k5,k6[,
    s1,s2,s3,s4[,tau_x,tau_y]]]]. Fisheye coefficients are not supported.
    """

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
                raise ValueError("Camera width and height must be positive integers")
        if not all(math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("Camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera focal lengths must be greater than zero")
        if self.distortion_coefficients is not None:
            coefficients = tuple(self.distortion_coefficients)
            if len(coefficients) not in (4, 5, 8, 12, 14):
                raise ValueError("OpenCV distortion requires 4, 5, 8, 12 or 14 coefficients")
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("Distortion coefficients must be finite")
            object.__setattr__(self, "distortion_coefficients", coefficients)

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
    ) -> "CameraIntrinsics":
        """Approximate a pinhole camera from known FOVs, without lens distortion."""
        for angle in (horizontal_fov_deg, vertical_fov_deg):
            if not 0 < angle < 180:
                raise ValueError("Camera FOV must be between 0 and 180 degrees")
        return cls(
            fx=width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2)),
            fy=height / (2 * math.tan(math.radians(vertical_fov_deg) / 2)),
            cx=width / 2,
            cy=height / 2,
            width=width,
            height=height,
        )


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Camera mounting Euler angles in degrees, relative to a forward camera.

    Zero mount: optical +Z = body Forward, image +X = body Right, image +Y =
    body Down. Apply Rz(yaw) @ Ry(pitch) @ Rx(roll) in FRD coordinates to
    that basis (right-handed active rotations). Pitch -90, roll/yaw 0 gives
    nadir with image top facing body Forward. No mount is assumed by default.
    """

    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        _validate_angles(self.roll_deg, self.pitch_deg, self.yaw_deg)


@dataclass(frozen=True, slots=True)
class TargetGeolocation:
    """Ground-plane target coordinate and horizontal offsets from the UAV."""

    coordinate: GeoCoordinate
    north_offset_m: float
    east_offset_m: float
    ground_distance_m: float
