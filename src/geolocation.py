"""Pure ground geometry: CAMERA Right/Down/Forward -> BODY FRD -> WORLD NED.

Column vectors use active right-handed rotations. No detection, grid, camera
backend, navigation decisions or flight-controller dependencies.
"""

import logging
import math

import numpy as np

from .models import (
    Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, Point,
    TargetGeolocation, UAVTelemetry,
)

logger = logging.getLogger(__name__)
WGS84_A_M = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def pixel_to_camera_ray(pixel: Point, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return a unit OpenCV ray. Input pixels must match calibration resolution."""
    if not (0 <= pixel.x < intrinsics.width and 0 <= pixel.y < intrinsics.height):
        raise ValueError("Pixel must be inside the calibrated image")
    if intrinsics.distortion_coefficients is None:
        x = (pixel.x - intrinsics.cx) / intrinsics.fx
        y = (pixel.y - intrinsics.cy) / intrinsics.fy
    else:
        import cv2

        matrix = np.array([[intrinsics.fx, 0, intrinsics.cx],
                           [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=float)
        points = np.array([[[pixel.x, pixel.y]]], dtype=float)
        x, y = cv2.undistortPoints(points, matrix,
                                  np.array(intrinsics.distortion_coefficients, dtype=float))[0, 0]
    ray = np.array([x, y, 1.0])
    norm = np.linalg.norm(ray)
    if not np.isfinite(ray).all() or not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid camera ray")
    return ray / norm


def _rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """ZYX Euler rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def camera_to_body_rotation(mount: CameraMount) -> np.ndarray:
    """Zero mount maps camera Right/Down/Forward to FRD Right/Down/Forward.

    Then apply mount Rz @ Ry @ Rx in FRD. Mount (0,-90,0) points down:
    image Right -> body Right, image Down -> body Backward, optical Z -> Down.
    """
    basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    return _rotation(mount.roll_deg, mount.pitch_deg, mount.yaw_deg) @ basis


def body_to_ned_rotation(attitude: Attitude) -> np.ndarray:
    """FRD to NED; zero attitude means Forward North, Right East, Down Down."""
    return _rotation(attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg)


def intersect_ground(ray_ned: np.ndarray, altitude_agl_m: float) -> tuple[float, float]:
    """Intersect forward ray with NED z=AGL, returning North/East meters.

    AGL is explicit ground clearance, never GNSS altitude. Ground is flat.
    Reject rays pointing upward or numerically too close to the horizon.
    """
    if not math.isfinite(altitude_agl_m) or altitude_agl_m <= 0:
        raise ValueError("altitude_agl_m must be finite and greater than zero")
    ray = np.asarray(ray_ned, dtype=float)
    if ray.shape != (3,) or not np.isfinite(ray).all():
        raise ValueError("A finite 3D ray is required")
    norm = np.linalg.norm(ray)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Ray must be nonzero")
    ray = ray / norm
    if ray[2] <= 1e-8:
        raise ValueError("Ray does not point toward ground or is too close to the horizon")
    north, east = altitude_agl_m / ray[2] * ray[:2]
    if not np.isfinite([north, east]).all():
        raise ValueError("Ground intersection is not finite")
    return float(north), float(east)


def offset_to_geodetic(origin: GeoCoordinate, north_m: float, east_m: float) -> GeoCoordinate:
    """Local WGS84 approximation with meridian and prime-vertical curvature.

    Intended for meters/tens of meters; reject offsets over 1 km and nonzero
    offsets within 1 degree of the poles. Uses the ellipsoid surface, not AGL.
    """
    if not all(math.isfinite(value) for value in (north_m, east_m)):
        raise ValueError("Offsets must be finite")
    if north_m == 0 and east_m == 0:
        return origin
    if math.hypot(north_m, east_m) > 1000 or abs(origin.latitude) >= 89:
        raise ValueError("Offset/location is outside the local WGS84 approximation limits")
    latitude = math.radians(origin.latitude)
    denominator = 1 - WGS84_E2 * math.sin(latitude) ** 2
    meridian = WGS84_A_M * (1 - WGS84_E2) / denominator ** 1.5
    prime_vertical = WGS84_A_M / math.sqrt(denominator)
    target_latitude = origin.latitude + math.degrees(north_m / meridian)
    longitude = origin.longitude + math.degrees(east_m / (prime_vertical * math.cos(latitude)))
    return GeoCoordinate(target_latitude, (longitude + 180) % 360 - 180)


class TargetGeolocator:
    """Compute the ground location of any input pixel, without confirming its identity."""

    def __init__(self, intrinsics: CameraIntrinsics, camera_mount: CameraMount) -> None:
        self.intrinsics, self.camera_mount = intrinsics, camera_mount

    def geolocate(self, pixel: Point, telemetry: UAVTelemetry) -> TargetGeolocation:
        ray_camera = pixel_to_camera_ray(pixel, self.intrinsics)
        ray_body = camera_to_body_rotation(self.camera_mount) @ ray_camera
        ray_ned = body_to_ned_rotation(telemetry.attitude) @ ray_body
        north, east = intersect_ground(ray_ned, telemetry.altitude_agl_m)
        logger.debug("Camera ray=%s; NED ray=%s; North=%+.3f East=%+.3f m",
                     ray_camera, ray_ned, north, east)
        return TargetGeolocation(offset_to_geodetic(telemetry.position, north, east),
                                  north, east, math.hypot(north, east))
