"""Pixel-to-ground geometry; no grid, hardware, navigation or flight commands.

Camera: Right, Down, Forward. Body: Forward, Right, Down (FRD).
World: North, East, Down (NED). Column vectors are actively rotated from
camera to body to NED. Ground is a flat plane z = altitude_agl_m in local NED.
"""

import math

import numpy as np

from .models import (
    Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, Point,
    TargetGeolocation, UAVTelemetry,
)

WGS84_A_M = 6378137.0
WGS84_FLATTENING = 1 / 298.257223563
WGS84_E2 = WGS84_FLATTENING * (2 - WGS84_FLATTENING)
MIN_DOWN_COMPONENT = 1e-8
MAX_LOCAL_OFFSET_M = 1000.0


def pixel_to_camera_ray(pixel: Point, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return a unit OpenCV camera ray (Right, Down, Forward).

    Pixels must use the calibration resolution, without unaccounted resizing,
    cropping or rotation. Correct optional pinhole distortion before K inverse.
    """
    if not (0 <= pixel.x < intrinsics.width and 0 <= pixel.y < intrinsics.height):
        raise ValueError("Pixel must lie inside the calibrated image")
    if intrinsics.distortion_coefficients is None:
        x = (pixel.x - intrinsics.cx) / intrinsics.fx
        y = (pixel.y - intrinsics.cy) / intrinsics.fy
    else:
        import cv2

        matrix = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        normalized = cv2.undistortPoints(
            np.array([[[pixel.x, pixel.y]]], dtype=np.float64),
            matrix,
            np.array(intrinsics.distortion_coefficients, dtype=np.float64),
        )
        x, y = normalized[0, 0]
    ray = np.array([x, y, 1.0], dtype=np.float64)
    length = np.linalg.norm(ray)
    if not np.isfinite(ray).all() or not math.isfinite(length) or length == 0:
        raise ValueError("Calibration produced an invalid camera ray")
    return ray / length


def _euler_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Right-handed active ZYX rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = map(math.radians, (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def camera_to_body_rotation(mount: CameraMount) -> np.ndarray:
    """Map OpenCV camera vectors to FRD using the explicit mount definition.

    At zero mount, camera Right -> body Right, Down -> Down, Forward -> Forward.
    Mount Euler rotation then acts on this FRD basis. A (0, -90, 0) mount maps
    camera Right -> body Right, Down -> body Backward, Forward -> body Down.
    """
    forward_camera_basis = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    return _euler_rotation(mount.roll_deg, mount.pitch_deg, mount.yaw_deg) @ forward_camera_basis


def body_to_ned_rotation(attitude: Attitude) -> np.ndarray:
    """Map FRD to NED; zero attitude means Forward North, Right East, Down Down.

    Positive yaw rotates North toward East; roll and pitch follow Attitude.
    External angles are degrees, yaw must reference true North.
    """
    return _euler_rotation(attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg)


def intersect_ground(ray_ned: np.ndarray, altitude_agl_m: float) -> tuple[float, float]:
    """Intersect a forward ray from (0,0,0) with NED z = positive AGL.

    Returns (North, East) in meters. Rejects upward/near-horizontal rays.
    Ray length is irrelevant; direction is normalized before horizon checking.
    """
    if not math.isfinite(altitude_agl_m) or altitude_agl_m <= 0:
        raise ValueError("altitude_agl_m must be finite and greater than zero")
    ray = np.asarray(ray_ned, dtype=float)
    if ray.shape != (3,) or not np.isfinite(ray).all():
        raise ValueError("Ground intersection requires a finite 3D ray")
    length = np.linalg.norm(ray)
    if not math.isfinite(length) or length == 0:
        raise ValueError("Ground intersection requires a nonzero ray")
    ray = ray / length
    if ray[2] <= MIN_DOWN_COMPONENT:
        raise ValueError("Ray does not point toward ground or is too close to the horizon")
    distance = altitude_agl_m / ray[2]
    north, east = distance * ray[:2]
    if not np.isfinite([north, east]).all():
        raise ValueError("Ground intersection is not finite")
    return float(north), float(east)


def offset_to_geodetic(origin: GeoCoordinate, north_m: float, east_m: float) -> GeoCoordinate:
    """First-order local WGS84 conversion using meridian/prime-vertical radii.

    Intended for meters to tens of meters; reject offsets beyond 1 km and
    nonzero offsets within 1 degree of a pole. Ellipsoid surface approximation,
    not a long-distance geodesic. Longitude wraps across the antimeridian.
    """
    if not all(math.isfinite(value) for value in (north_m, east_m)):
        raise ValueError("North/East offsets must be finite")
    if north_m == 0 and east_m == 0:
        return origin
    if math.hypot(north_m, east_m) > MAX_LOCAL_OFFSET_M:
        raise ValueError("Offset exceeds the 1 km local WGS84 approximation limit")
    if abs(origin.latitude) >= 89:
        raise ValueError("Local WGS84 approximation is not supported near the poles")
    latitude = math.radians(origin.latitude)
    denominator = 1 - WGS84_E2 * math.sin(latitude) ** 2
    meridian_radius = WGS84_A_M * (1 - WGS84_E2) / denominator ** 1.5
    prime_vertical_radius = WGS84_A_M / math.sqrt(denominator)
    target_latitude = origin.latitude + math.degrees(north_m / meridian_radius)
    target_longitude = origin.longitude + math.degrees(
        east_m / (prime_vertical_radius * math.cos(latitude))
    )
    target_longitude = (target_longitude + 180) % 360 - 180
    return GeoCoordinate(latitude=target_latitude, longitude=target_longitude)


class TargetGeolocator:
    """Locate pixels on flat ground using explicit calibration, mount and AGL."""

    def __init__(self, intrinsics: CameraIntrinsics, camera_mount: CameraMount) -> None:
        self.intrinsics = intrinsics
        self.camera_mount = camera_mount

    def geolocate(self, pixel: Point, telemetry: UAVTelemetry) -> TargetGeolocation:
        """Compute a ground coordinate, independently of any field grid."""
        ray_camera = pixel_to_camera_ray(pixel, self.intrinsics)
        ray_body = camera_to_body_rotation(self.camera_mount) @ ray_camera
        ray_ned = body_to_ned_rotation(telemetry.attitude) @ ray_body
        north, east = intersect_ground(ray_ned, telemetry.altitude_agl_m)
        coordinate = offset_to_geodetic(telemetry.position, north, east)
        return TargetGeolocation(
            coordinate=coordinate,
            north_offset_m=north,
            east_offset_m=east,
            ground_distance_m=math.hypot(north, east),
        )
