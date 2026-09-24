"""Analytic geometry tests without camera or UAV hardware."""

import math
import unittest
from dataclasses import replace

import numpy as np

from src.geolocation import (
    TargetGeolocator, body_to_ned_rotation, camera_to_body_rotation,
    intersect_ground, offset_to_geodetic, pixel_to_camera_ray,
)
from src.models import (
    Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, Point, UAVTelemetry,
)
from src.telemetry import StaticTelemetryProvider


# Synthetic calibration and mount; these are not production camera parameters.
INTRINSICS = CameraIntrinsics(200, 200, 320, 240, 640, 480)
NADIR = CameraMount(0, -90, 0)
TELEMETRY = UAVTelemetry(GeoCoordinate(50, 14), 10, Attitude(0, 0, 0))


class CameraGeometryTests(unittest.TestCase):
    def test_principal_point_ray(self) -> None:
        np.testing.assert_allclose(pixel_to_camera_ray(Point(320, 240), INTRINSICS), [0, 0, 1])

    def test_off_center_ray_is_normalized(self) -> None:
        ray = pixel_to_camera_ray(Point(420, 340), INTRINSICS)
        np.testing.assert_allclose(ray, np.array([0.5, 0.5, 1]) / math.sqrt(1.5))
        self.assertAlmostEqual(np.linalg.norm(ray), 1)

    def test_distortion_is_removed(self) -> None:
        # Ideal x/z=.5, r^2=.25, k1=.2 => distorted x=.525 => u=425.
        calibration = replace(INTRINSICS, distortion_coefficients=(0.2, 0, 0, 0, 0))
        ray = pixel_to_camera_ray(Point(425, 240), calibration)
        self.assertAlmostEqual(ray[0] / ray[2], 0.5, places=6)
        self.assertAlmostEqual(ray[1], 0)

    def test_zero_distortion_matches_pinhole(self) -> None:
        for size in (4, 5, 8, 12, 14):
            with self.subTest(size=size):
                calibration = replace(INTRINSICS, distortion_coefficients=(0.0,) * size)
                np.testing.assert_allclose(
                    pixel_to_camera_ray(Point(410, 300), calibration),
                    pixel_to_camera_ray(Point(410, 300), INTRINSICS),
                )

    def test_fov_constructor(self) -> None:
        calibration = CameraIntrinsics.from_fov(800, 600, 90, 90)
        self.assertAlmostEqual(calibration.fx, 400)
        self.assertAlmostEqual(calibration.fy, 300)
        self.assertEqual((calibration.cx, calibration.cy), (400, 300))

    def test_invalid_calibration_and_fov(self) -> None:
        for values in (
            {"fx": 0}, {"fy": -1}, {"cx": float("nan")}, {"fx": float("inf")},
            {"width": 0}, {"height": 1.5}, {"width": True},
            {"distortion_coefficients": (0, 0)},
            {"distortion_coefficients": (0, 0, 0, float("nan"))},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(INTRINSICS, **values)
        for angle in (0, -1, 180, 181, float("nan"), float("inf")):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                CameraIntrinsics.from_fov(640, 480, angle, 60)

    def test_invalid_pixels(self) -> None:
        for point in (Point(-1, 0), Point(640, 0), Point(0, 480), Point(float("nan"), 0)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                pixel_to_camera_ray(point, INTRINSICS)

    def test_forward_and_nadir_axes(self) -> None:
        forward = camera_to_body_rotation(CameraMount(0, 0, 0))
        np.testing.assert_allclose(forward @ [0, 0, 1], [1, 0, 0])
        nadir = camera_to_body_rotation(NADIR)
        np.testing.assert_allclose(nadir @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(nadir @ [0, 1, 0], [-1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(nadir @ [0, 0, 1], [0, 0, 1], atol=1e-12)

    def test_rotations_are_proper_and_yaw_turns_east(self) -> None:
        for matrix in (
            camera_to_body_rotation(CameraMount(12, -73, 23)),
            body_to_ned_rotation(Attitude(17, 31, 87)),
        ):
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(matrix), 1)
        np.testing.assert_allclose(
            body_to_ned_rotation(Attitude(0, 0, 90)) @ [1, 0, 0], [0, 1, 0], atol=1e-12
        )

    def test_invalid_angles(self) -> None:
        for cls in (Attitude, CameraMount):
            for angles in ((float("nan"), 0, 0), (0, float("inf"), 0), (0, 0, float("nan"))):
                with self.subTest(cls=cls, angles=angles), self.assertRaises(ValueError):
                    cls(*angles)


class TargetGeolocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geolocator = TargetGeolocator(INTRINSICS, NADIR)

    def test_nadir_principal_point_matches_uav(self) -> None:
        result = self.geolocator.geolocate(Point(320, 240), TELEMETRY)
        self.assertAlmostEqual(result.north_offset_m, 0)
        self.assertAlmostEqual(result.east_offset_m, 0)
        self.assertAlmostEqual(result.ground_distance_m, 0)
        self.assertAlmostEqual(result.coordinate.latitude, 50)
        self.assertAlmostEqual(result.coordinate.longitude, 14)

    def test_image_directions(self) -> None:
        for pixel, north, east in (
            (Point(420, 240), 0, 5), (Point(220, 240), 0, -5),
            (Point(320, 140), 5, 0), (Point(320, 340), -5, 0),
        ):
            with self.subTest(pixel=pixel):
                result = self.geolocator.geolocate(pixel, TELEMETRY)
                self.assertAlmostEqual(result.north_offset_m, north)
                self.assertAlmostEqual(result.east_offset_m, east)
                self.assertAlmostEqual(result.ground_distance_m, 5)

    def test_yaw_90_rotates_right_pixel_to_south(self) -> None:
        telemetry = replace(TELEMETRY, attitude=Attitude(0, 0, 90))
        result = self.geolocator.geolocate(Point(420, 240), telemetry)
        self.assertAlmostEqual(result.north_offset_m, -5)
        self.assertAlmostEqual(result.east_offset_m, 0)
        result = self.geolocator.geolocate(Point(320, 140), telemetry)
        self.assertAlmostEqual(result.north_offset_m, 0)
        self.assertAlmostEqual(result.east_offset_m, 5)

    def test_roll_and_pitch(self) -> None:
        offset = 10 * math.tan(math.radians(30))
        for attitude, north, east in (
            (Attitude(0, 30, 0), offset, 0),
            (Attitude(0, -30, 0), -offset, 0),
            (Attitude(30, 0, 0), 0, -offset),
            (Attitude(-30, 0, 0), 0, offset),
            (Attitude(0, 30, 90), 0, offset),
        ):
            with self.subTest(attitude=attitude):
                result = self.geolocator.geolocate(
                    Point(320, 240), replace(TELEMETRY, attitude=attitude)
                )
                self.assertAlmostEqual(result.north_offset_m, north)
                self.assertAlmostEqual(result.east_offset_m, east)

    def test_mount_is_configurable(self) -> None:
        geolocator = TargetGeolocator(INTRINSICS, CameraMount(0, -45, 0))
        result = geolocator.geolocate(Point(320, 240), TELEMETRY)
        self.assertAlmostEqual(result.north_offset_m, 10)
        self.assertAlmostEqual(result.east_offset_m, 0)
        rotated = TargetGeolocator(INTRINSICS, CameraMount(0, -90, 90))
        result = rotated.geolocate(Point(420, 240), TELEMETRY)
        self.assertAlmostEqual(result.north_offset_m, -5)

    def test_altitude_scales_offset(self) -> None:
        result = self.geolocator.geolocate(Point(420, 240), replace(TELEMETRY, altitude_agl_m=20))
        self.assertAlmostEqual(result.east_offset_m, 10)

    def test_upward_and_horizontal_mounts_are_rejected(self) -> None:
        for mount in (CameraMount(0, 0, 0), CameraMount(0, 90, 0)):
            with self.subTest(mount=mount), self.assertRaises(ValueError):
                TargetGeolocator(INTRINSICS, mount).geolocate(Point(320, 240), TELEMETRY)

    def test_invalid_rays(self) -> None:
        for ray in ([0, 0, -1], [1, 0, 0], [1, 0, 1e-12], [0, 0, 0],
                    [0, float("nan"), 1], [1, 2], [float("inf"), 0, 1]):
            with self.subTest(ray=ray), self.assertRaises(ValueError):
                intersect_ground(np.array(ray), 10)
        self.assertEqual(intersect_ground(np.array([2, 3, 2]), 10), (10, 15))

    def test_invalid_altitude(self) -> None:
        for altitude in (0, -1, float("nan"), float("inf")):
            with self.subTest(altitude=altitude):
                with self.assertRaises(ValueError):
                    replace(TELEMETRY, altitude_agl_m=altitude)
                with self.assertRaises(ValueError):
                    intersect_ground(np.array([0, 0, 1]), altitude)

    def test_static_provider_requires_explicit_sample(self) -> None:
        provider = StaticTelemetryProvider(TELEMETRY)
        self.assertIs(provider.get_telemetry(), TELEMETRY)
        with self.assertRaises(TypeError):
            StaticTelemetryProvider()


class WGS84Tests(unittest.TestCase):
    def test_equatorial_reference_distances(self) -> None:
        # WGS84: meridional degree ~110574.276 m, equatorial degree ~111319.491 m.
        origin = GeoCoordinate(0, 0)
        north = offset_to_geodetic(origin, 10, 0)
        east = offset_to_geodetic(origin, 0, 10)
        self.assertAlmostEqual(north.latitude, 0.000090436947705, places=12)
        self.assertAlmostEqual(east.longitude, 0.000089831528412, places=12)
        self.assertEqual(north.longitude, 0)
        self.assertEqual(east.latitude, 0)

    def test_midlatitude_reference_and_signs(self) -> None:
        origin = GeoCoordinate(50, 14)
        target = offset_to_geodetic(origin, 10, 10)
        self.assertAlmostEqual(target.latitude, 50.0000899046, places=9)
        self.assertAlmostEqual(target.longitude, 14.0001394783, places=9)
        southwest = offset_to_geodetic(origin, -10, -10)
        self.assertLess(southwest.latitude, origin.latitude)
        self.assertLess(southwest.longitude, origin.longitude)

    def test_zero_offset(self) -> None:
        origin = GeoCoordinate(50, 14)
        self.assertEqual(offset_to_geodetic(origin, 0, 0), origin)

    def test_antimeridian(self) -> None:
        self.assertLess(offset_to_geodetic(GeoCoordinate(0, 179.99999), 0, 10).longitude, -179.99)
        self.assertGreater(offset_to_geodetic(GeoCoordinate(0, -179.99999), 0, -10).longitude, 179.99)

    def test_invalid_coordinates_offsets_and_local_limits(self) -> None:
        for latitude, longitude in ((91, 0), (0, 181), (float("nan"), 0), (0, float("inf"))):
            with self.subTest(latitude=latitude), self.assertRaises(ValueError):
                GeoCoordinate(latitude, longitude)
        for origin, north, east in (
            (GeoCoordinate(50, 14), float("nan"), 0),
            (GeoCoordinate(50, 14), 0, float("inf")),
            (GeoCoordinate(50, 14), 1001, 0),
            (GeoCoordinate(90, 0), 0, 1),
        ):
            with self.subTest(origin=origin, north=north, east=east), self.assertRaises(ValueError):
                offset_to_geodetic(origin, north, east)


if __name__ == "__main__":
    unittest.main()
