"""Analytic geolocation checks using explicit synthetic calibration/telemetry."""

import math
import unittest
from dataclasses import replace

import numpy as np

from src.geolocation import (
    TargetGeolocator, body_to_ned_rotation, camera_to_body_rotation,
    intersect_ground, offset_to_geodetic, pixel_to_camera_ray,
)
from src.models import Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, Point, UAVTelemetry
from src.telemetry import telemetry_from_dict

INTRINSICS = CameraIntrinsics(200, 200, 320, 240, 640, 480)
TELEMETRY = UAVTelemetry(GeoCoordinate(50, 14), 10, Attitude(0, 0, 0))


class GeolocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.locator = TargetGeolocator(INTRINSICS, CameraMount(0, -90, 0))

    def test_center_pixel(self) -> None:
        result = self.locator.geolocate(Point(320, 240), TELEMETRY)
        self.assertAlmostEqual(result.north_offset_m, 0)
        self.assertAlmostEqual(result.east_offset_m, 0)
        self.assertAlmostEqual(result.coordinate.latitude, 50)
        self.assertAlmostEqual(result.coordinate.longitude, 14)

    def test_pixel_directions(self) -> None:
        for pixel, north, east in ((Point(420, 240), 0, 5), (Point(220, 240), 0, -5),
                                    (Point(320, 140), 5, 0), (Point(320, 340), -5, 0)):
            with self.subTest(pixel=pixel):
                result = self.locator.geolocate(pixel, TELEMETRY)
                self.assertAlmostEqual(result.north_offset_m, north)
                self.assertAlmostEqual(result.east_offset_m, east)
                self.assertAlmostEqual(result.ground_distance_m, 5)

    def test_yaw(self) -> None:
        telemetry = replace(TELEMETRY, attitude=Attitude(0, 0, 90))
        result = self.locator.geolocate(Point(420, 240), telemetry)
        self.assertAlmostEqual(result.north_offset_m, -5)
        self.assertAlmostEqual(result.east_offset_m, 0)

    def test_roll_pitch(self) -> None:
        offset = 10 * math.tan(math.radians(30))
        for attitude, north, east in ((Attitude(30, 0, 0), 0, -offset),
                                      (Attitude(-30, 0, 0), 0, offset),
                                      (Attitude(0, 30, 0), offset, 0),
                                      (Attitude(0, -30, 0), -offset, 0)):
            with self.subTest(attitude=attitude):
                result = self.locator.geolocate(Point(320, 240), replace(TELEMETRY, attitude=attitude))
                self.assertAlmostEqual(result.north_offset_m, north)
                self.assertAlmostEqual(result.east_offset_m, east)

    def test_configurable_mount(self) -> None:
        locator = TargetGeolocator(INTRINSICS, CameraMount(0, -45, 0))
        result = locator.geolocate(Point(320, 240), TELEMETRY)
        self.assertAlmostEqual(result.north_offset_m, 10)
        self.assertAlmostEqual(result.east_offset_m, 0)

    def test_coordinate_bases_and_rotation(self) -> None:
        nadir = camera_to_body_rotation(CameraMount(0, -90, 0))
        np.testing.assert_allclose(nadir @ [0, 0, 1], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(nadir @ [0, 1, 0], [-1, 0, 0], atol=1e-12)
        rotation = body_to_ned_rotation(Attitude(20, -10, 80))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_invalid_ray(self) -> None:
        for ray in ([0, 0, -1], [1, 0, 0], [1, 0, 1e-12], [0, 0, 0], [float("nan"), 0, 1]):
            with self.subTest(ray=ray), self.assertRaises(ValueError):
                intersect_ground(np.array(ray), 10)

    def test_invalid_altitude_and_scaling(self) -> None:
        for altitude in (0, -1, float("nan"), float("inf")):
            with self.subTest(altitude=altitude):
                with self.assertRaises(ValueError):
                    replace(TELEMETRY, altitude_agl_m=altitude)
                with self.assertRaises(ValueError):
                    intersect_ground(np.array([0, 0, 1]), altitude)
        result = self.locator.geolocate(Point(420, 240), replace(TELEMETRY, altitude_agl_m=20))
        self.assertAlmostEqual(result.east_offset_m, 10)

    def test_intrinsics_fov_and_distortion(self) -> None:
        calibration = CameraIntrinsics.from_fov(800, 600, 90, 90)
        self.assertAlmostEqual(calibration.fx, 400)
        self.assertAlmostEqual(calibration.fy, 300)
        np.testing.assert_allclose(pixel_to_camera_ray(Point(320, 240), INTRINSICS), [0, 0, 1])
        zero = replace(INTRINSICS, distortion_coefficients=(0, 0, 0, 0, 0))
        np.testing.assert_allclose(pixel_to_camera_ray(Point(420, 240), zero),
                                   pixel_to_camera_ray(Point(420, 240), INTRINSICS))
        distorted = replace(INTRINSICS, distortion_coefficients=(0.2, 0, 0, 0, 0))
        ray = pixel_to_camera_ray(Point(425, 240), distorted)
        self.assertAlmostEqual(ray[0] / ray[2], 0.5, places=6)

    def test_invalid_calibration_and_pixel(self) -> None:
        for values in ({"fx": 0}, {"height": 0}, {"cx": float("nan")},
                       {"distortion_coefficients": (0, 0)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(INTRINSICS, **values)
        with self.assertRaises(ValueError):
            pixel_to_camera_ray(Point(640, 480), INTRINSICS)

    def test_wgs84_reference_values(self) -> None:
        origin = GeoCoordinate(0, 0)
        north = offset_to_geodetic(origin, 10, 0)
        east = offset_to_geodetic(origin, 0, 10)
        self.assertAlmostEqual(north.latitude, 0.000090436947705, places=12)
        self.assertAlmostEqual(east.longitude, 0.000089831528412, places=12)
        target = offset_to_geodetic(GeoCoordinate(50, 14), 10, 10)
        self.assertAlmostEqual(target.latitude, 50.0000899046, places=9)
        self.assertAlmostEqual(target.longitude, 14.0001394783, places=9)
        self.assertEqual(offset_to_geodetic(origin, 0, 0), origin)

    def test_antimeridian_and_invalid_offsets(self) -> None:
        self.assertLess(offset_to_geodetic(GeoCoordinate(0, 179.99999), 0, 10).longitude, -179.99)
        for origin, north, east in ((GeoCoordinate(90, 0), 1, 0), (GeoCoordinate(0, 0), 1001, 0),
                                    (GeoCoordinate(0, 0), float("nan"), 0)):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                offset_to_geodetic(origin, north, east)

    def test_telemetry_never_substitutes_gnss_altitude(self) -> None:
        data = {"position": {"latitude": 50, "longitude": 14}, "altitude": 500,
                "attitude": {"roll_deg": 0, "pitch_deg": 0, "yaw_deg": 0}}
        with self.assertRaises(KeyError):
            telemetry_from_dict(data)


if __name__ == "__main__":
    unittest.main()
