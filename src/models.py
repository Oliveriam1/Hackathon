"""Shared geometry and mission data. Distances are meters and angles degrees."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True, slots=True)
class Point:
    """Image pixel coordinates (u, v)."""
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LocalPoint:
    """Ground plane: x East, y North, in meters from a documented origin."""
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x, self.y)):
            raise ValueError("Local coordinates must be finite")


@dataclass(frozen=True, slots=True)
class GeoCoordinate:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("Invalid WGS84 coordinate")


@dataclass(frozen=True, slots=True)
class Attitude:
    """FRD body to NED: Rz(yaw) Ry(pitch) Rx(roll); true-North yaw."""
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.roll_deg, self.pitch_deg, self.yaw_deg)):
            raise ValueError("Attitude must be finite")


@dataclass(frozen=True, slots=True)
class UAVTelemetry:
    position: GeoCoordinate
    altitude_agl_m: float
    attitude: Attitude

    def __post_init__(self) -> None:
        if not math.isfinite(self.altitude_agl_m) or self.altitude_agl_m < 0:
            raise ValueError("AGL must be finite and nonnegative; geolocation requires AGL > 0")


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("Camera dimensions must be positive integers")
        if not all(math.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)) or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid camera intrinsics")

    @classmethod
    def from_fov(cls, width: int, height: int, horizontal_fov_deg: float,
                 vertical_fov_deg: float) -> CameraIntrinsics:
        if not (0 < horizontal_fov_deg < 180 and 0 < vertical_fov_deg < 180):
            raise ValueError("Both FOV angles must be between 0 and 180 degrees")
        return cls(width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2)),
                   height / (2 * math.tan(math.radians(vertical_fov_deg) / 2)),
                   width / 2, height / 2, width, height)


@dataclass(frozen=True, slots=True)
class CameraMount:
    """Rz Ry Rx relative to a forward camera. (0,-90,0): nadir, image top forward."""
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.roll_deg, self.pitch_deg, self.yaw_deg)):
            raise ValueError("Camera mount must be finite")


@dataclass(frozen=True, slots=True)
class TargetLocation:
    coordinate: GeoCoordinate
    north_offset_m: float
    east_offset_m: float


class FlightAction(Enum):
    TAKEOFF = "TAKEOFF"
    GOTO = "GOTO"
    ASCEND = "ASCEND"


@dataclass(frozen=True, slots=True)
class FlightCommand:
    """High-level intent; an adapter must wait for confirmed completion or fail."""
    action: FlightAction
    position: GeoCoordinate
    altitude_agl_m: float
    yaw_deg: float


class MissionState(Enum):
    INIT = "INIT"
    MOVING_TO_B = "MOVING_TO_B"
    CALCULATING_CENTER = "CALCULATING_CENTER"
    ASCENDING = "ASCENDING"
    SCANNING = "SCANNING"
    TARGET_FOUND = "TARGET_FOUND"
    SENDING_DATA = "SENDING_DATA"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
