"""Pixel/gimbal -> viewing angle -> ground distance; pure math, no hardware.

Angles are degrees. ``angle_from_down_deg`` is measured from the vertical DOWN
axis, not from the horizon and NOT from an object's contour/ellipse rotation.
Ground distance = AGL * tan(angle); slant distance = AGL / cos(angle).
Equivalently ground distance = slant distance * sin(angle), never AGL*sin(angle).

Gimbal convention matches red_tracker.SimCamera: zero points down, image top
faces forward; positive X tilts right and positive Y tilts forward. Body attitude
uses FRD: positive roll lowers the right wing, positive pitch raises the nose.
Bearings are clockwise from the aircraft's horizontal heading (right = +90).
Use heading/yaw separately for geographic coordinates. At nadir bearing is None.

Requires calibrated, distortion-corrected pixels, measured gimbal orientation and
attitude for the same frame, and altitude ABOVE THE TARGET GROUND PLANE (AGL),
not GNSS/MSL height or slant range. Assumes flat ground and collocated camera/UAV.
No telemetry defaults or camera FOV are invented here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from typing import Mapping

from src.models import CameraIntrinsics, Point


@dataclass(frozen=True, slots=True)
class ViewingAngles:
    angle_from_down_deg: float
    bearing_from_heading_deg: float | None

    def __post_init__(self) -> None:
        if (not math.isfinite(self.angle_from_down_deg) or not 0 <= self.angle_from_down_deg < 90
                or math.cos(math.radians(self.angle_from_down_deg)) <= 1e-8):
            raise ValueError("Viewing ray must point down and be away from the horizon")
        if self.bearing_from_heading_deg is not None and not math.isfinite(self.bearing_from_heading_deg):
            raise ValueError("Bearing must be finite")
        if self.angle_from_down_deg > 0 and self.bearing_from_heading_deg is None:
            raise ValueError("An off-nadir ray requires a bearing")


@dataclass(frozen=True, slots=True)
class GroundMeasurement:
    angles: ViewingAngles
    altitude_agl_m: float
    ground_distance_m: float
    slant_distance_m: float
    right_offset_m: float
    forward_offset_m: float


def pixel_to_viewing_angles(pixel: Point, intrinsics: CameraIntrinsics, *,
                           gimbal_right_deg: float, gimbal_forward_deg: float,
                           roll_deg: float, pitch_deg: float) -> ViewingAngles:
    """Compose camera and FRD rotations; no additive pixel-angle approximation.

    Camera pixel axes: right/down; optical axis forward. Rotate the gimballed
    camera into body FRD, then apply Ry(pitch) Rx(roll) to a level heading frame.
    Yaw is omitted deliberately: bearing is relative to heading, not North.
    """
    values = (pixel.x, pixel.y, gimbal_right_deg, gimbal_forward_deg, roll_deg, pitch_deg)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Pixel and orientation must be finite")
    if not (0 <= pixel.x < intrinsics.width and 0 <= pixel.y < intrinsics.height):
        raise ValueError("Pixel is outside the calibrated image")
    x, y = (pixel.x - intrinsics.cx) / intrinsics.fx, (pixel.y - intrinsics.cy) / intrinsics.fy
    ax, ay, roll, pitch = map(math.radians, (gimbal_right_deg, gimbal_forward_deg, roll_deg, pitch_deg))
    sx, cx, sy, cy = math.sin(ax), math.cos(ax), math.sin(ay), math.cos(ay)
    # Same right/up/optical basis as red_tracker.SimCamera._project, expressed FRD.
    forward = sy - cy * y
    right = cx * x + sx * sy * y + cy * sx
    down = -sx * x + sy * cx * y + cy * cx
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rolled_right, rolled_down = cr * right - sr * down, sr * right + cr * down
    forward, down = cp * forward + sp * rolled_down, -sp * forward + cp * rolled_down
    right = rolled_right
    norm = math.sqrt(forward * forward + right * right + down * down)
    if not math.isfinite(norm) or norm == 0 or down / norm <= 1e-8:
        raise ValueError("Viewing ray is above or too close to the horizon")
    horizontal = math.hypot(right, forward)
    if horizontal / norm < 1e-12:
        return ViewingAngles(0.0, None)
    return ViewingAngles(math.degrees(math.atan2(horizontal, down)),
                         math.degrees(math.atan2(right, forward)))


def distance_from_angles(angles: ViewingAngles, altitude_agl_m: float) -> GroundMeasurement:
    """Distance from the point directly BELOW the UAV to the target on flat ground."""
    if not math.isfinite(altitude_agl_m) or altitude_agl_m <= 0:
        raise ValueError("altitude_agl_m must be positive and finite")
    theta = math.radians(angles.angle_from_down_deg)
    ground = altitude_agl_m * math.tan(theta)
    slant = altitude_agl_m / math.cos(theta)
    if not math.isfinite(ground) or not math.isfinite(slant):
        raise ValueError("Computed distance is not finite")
    bearing = math.radians(angles.bearing_from_heading_deg or 0.0)
    return GroundMeasurement(angles, altitude_agl_m, ground, slant,
                             ground * math.sin(bearing), ground * math.cos(bearing))


def from_red_detection(detection: Mapping[str, float] | None, *, intrinsics: CameraIntrinsics,
                       frame_size: tuple[int, int], altitude_agl_m: float,
                       gimbal_right_deg: float, gimbal_forward_deg: float,
                       roll_deg: float, pitch_deg: float) -> GroundMeasurement | None:
    """Adapter for red_tracker.detect's dict; only x/y are used, never tilt/angle.

    frame_size is (width, height) of the ORIGINAL image, not a resized preview.
    Missing detection returns None; malformed geometry/calibration raises ValueError.
    Supply the measured gimbal orientation at exposure, not a later servo command.
    """
    if detection is None:
        return None
    if frame_size != (intrinsics.width, intrinsics.height):
        raise ValueError("Frame resolution does not match camera calibration")
    try:
        pixel = Point(float(detection["x"]), float(detection["y"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Detection must provide finite pixel x and y") from exc
    angles = pixel_to_viewing_angles(pixel, intrinsics, gimbal_right_deg=gimbal_right_deg,
                                    gimbal_forward_deg=gimbal_forward_deg,
                                    roll_deg=roll_deg, pitch_deg=pitch_deg)
    return distance_from_angles(angles, altitude_agl_m)


def main() -> int:
    parser = argparse.ArgumentParser(description="Viewing angle from vertical + AGL -> ground distance (JSON)")
    parser.add_argument("--altitude-agl", type=float, required=True, help="Height above target ground in metres")
    parser.add_argument("--angle-deg", type=float, required=True, help="Viewing angle from vertical down")
    parser.add_argument("--bearing-deg", type=float, help="Clockwise bearing from heading, right=90")
    args = parser.parse_args()
    try:
        result = distance_from_angles(ViewingAngles(args.angle_deg, args.bearing_deg), args.altitude_agl)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
