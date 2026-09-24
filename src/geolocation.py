"""Pinhole ray -> flat ground -> local WGS84. No hardware, detection or control."""

import math
import numpy as np

from .models import CameraIntrinsics, CameraMount, GeoCoordinate, LocalPoint, Point, TargetLocation, UAVTelemetry

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def _radii(origin: GeoCoordinate) -> tuple[float, float]:
    if abs(origin.latitude) >= 89:
        raise ValueError("Local WGS84 approximation is unsupported within 1 degree of the poles")
    lat = math.radians(origin.latitude)
    d = 1 - E2 * math.sin(lat) ** 2
    return A * (1 - E2) / d ** 1.5, A / math.sqrt(d) * math.cos(lat)


def geodetic_to_local(origin: GeoCoordinate, point: GeoCoordinate) -> LocalPoint:
    """Short-distance x=East/y=North coordinates relative to origin."""
    meridian, parallel = _radii(origin)
    lon_delta = (point.longitude - origin.longitude + 180) % 360 - 180
    local = LocalPoint(math.radians(lon_delta) * parallel,
                       math.radians(point.latitude - origin.latitude) * meridian)
    if math.hypot(local.x, local.y) > 1000:
        raise ValueError("This local map supports distances up to 1 km")
    return local


def local_to_geodetic(origin: GeoCoordinate, point: LocalPoint) -> GeoCoordinate:
    if math.hypot(point.x, point.y) > 1000:
        raise ValueError("This local map supports distances up to 1 km")
    meridian, parallel = _radii(origin)
    return GeoCoordinate(origin.latitude + math.degrees(point.y / meridian),
                         (origin.longitude + math.degrees(point.x / parallel) + 180) % 360 - 180)


def _rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def geolocate(pixel: Point, telemetry: UAVTelemetry, intrinsics: CameraIntrinsics,
               mount: CameraMount) -> TargetLocation:
    """CAMERA Right/Down/Forward -> BODY Forward/Right/Down -> NED North/East/Down.

    Active column-vector rotations are Rz(yaw) Ry(pitch) Rx(roll). Zero mount
    looks forward; pitch -90 is nadir. Ground is z=AGL in NED. AGL is never
    inferred from GNSS height. Pixels must be distortion-corrected and match K.
    """
    if telemetry.altitude_agl_m <= 0:
        raise ValueError("Geolocation requires positive altitude_agl_m")
    if not (0 <= pixel.x < intrinsics.width and 0 <= pixel.y < intrinsics.height):
        raise ValueError("Pixel is outside the calibrated image")
    ray = np.array([(pixel.x - intrinsics.cx) / intrinsics.fx,
                    (pixel.y - intrinsics.cy) / intrinsics.fy, 1.0])
    ray /= np.linalg.norm(ray)
    camera_basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    attitude = telemetry.attitude
    ned = (_rotation(attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg)
           @ _rotation(mount.roll_deg, mount.pitch_deg, mount.yaw_deg) @ camera_basis @ ray)
    if not np.isfinite(ned).all() or ned[2] <= 1e-8:
        raise ValueError("Ray is above or too close to the horizon")
    north, east = telemetry.altitude_agl_m / ned[2] * ned[:2]
    offset = LocalPoint(float(east), float(north))
    return TargetLocation(local_to_geodetic(telemetry.position, offset), float(north), float(east))
