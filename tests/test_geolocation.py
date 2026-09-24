import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.camera_calibration import calibrate, find_corners
from src.geolocation import (CameraModel, DronePose, GimbalAngles, ServoAxis, Uncertainty, body_in_ned,
                             camera_in_body, centering_angles, combine, gimbal_from_servos, locate,
                             offset_to_latlon)

CAMERA = CameraModel.from_fov((640, 480), (54.0, 42.0))
CENTER = (320, 240)
LAT, LON = 50.0, 14.0


def pose(**values):
    return DronePose(**{'lat': LAT, 'lon': LON, 'height': 20.0, **values})


class GeometryTests(unittest.TestCase):
    def assertOffset(self, fix, north, east, delta=0.01):
        self.assertAlmostEqual(fix.north, north, delta=delta)
        self.assertAlmostEqual(fix.east, east, delta=delta)

    def test_straight_down_is_under_the_drone(self):
        fix = locate(CENTER, pose(), GimbalAngles(), CAMERA)
        self.assertOffset(fix, 0, 0)
        self.assertAlmostEqual(fix.lat, LAT, places=9)
        self.assertAlmostEqual(fix.lon, LON, places=9)

    def test_gimbal_angles_and_heading(self):
        # Kamera 45° dopředu z 20 m vidí bod 20 m před dronem.
        self.assertOffset(locate(CENTER, pose(yaw=0), GimbalAngles(forward=45), CAMERA), 20, 0)
        self.assertOffset(locate(CENTER, pose(yaw=90), GimbalAngles(forward=45), CAMERA), 0, 20)
        self.assertOffset(locate(CENTER, pose(yaw=180), GimbalAngles(forward=45), CAMERA), -20, 0)
        self.assertOffset(locate(CENTER, pose(yaw=0), GimbalAngles(right=30), CAMERA), 0, 20*math.tan(math.radians(30)))
        self.assertOffset(locate(CENTER, pose(yaw=90), GimbalAngles(right=30), CAMERA), -20*math.tan(math.radians(30)), 0)
        fix = locate(CENTER, pose(height=10), GimbalAngles(forward=60), CAMERA)
        self.assertAlmostEqual(fix.distance, 10*math.tan(math.radians(60)), delta=0.01)
        self.assertAlmostEqual(fix.off_nadir, 60, delta=0.01)
        self.assertAlmostEqual(fix.bearing, 0, delta=0.01)

    def test_outer_axis_carries_inner_axis(self):
        # Vnější x o 30° doprava, pak vnitřní y o 40° dopředu (osa y se naklonila s x).
        right, forward = math.radians(30), math.radians(40)
        axis = np.array([math.sin(forward), math.sin(right)*math.cos(forward), math.cos(right)*math.cos(forward)])
        fix = locate(CENTER, pose(), GimbalAngles(right=30, forward=40), CAMERA)
        self.assertOffset(fix, 20*axis[0]/axis[2], 20*axis[1]/axis[2])

    def test_pixel_offsets_follow_image_orientation(self):
        fx, fy = CAMERA.matrix[0, 0], CAMERA.matrix[1, 1]
        step = math.tan(math.radians(10))
        right_pixel = (320 + fx*step, 240)
        up_pixel = (320, 240 - fy*step)
        self.assertOffset(locate(right_pixel, pose(), GimbalAngles(), CAMERA), 0, 20*step)
        self.assertOffset(locate(up_pixel, pose(), GimbalAngles(), CAMERA), 20*step, 0)
        # Horní okraj obrazu míří doprava: obraz nahoru = východ, obraz vpravo = jih.
        self.assertOffset(locate(up_pixel, pose(), GimbalAngles(), CAMERA, image_top='right'), 0, 20*step)
        self.assertOffset(locate(right_pixel, pose(), GimbalAngles(), CAMERA, image_top='right'), -20*step, 0)

    def test_drone_attitude_is_compensated(self):
        # Pravé rameno dolů o 10° naklání kameru doleva, závěs 10° doprava to vyrovná.
        self.assertOffset(locate(CENTER, pose(roll=10), GimbalAngles(right=10), CAMERA), 0, 0)
        self.assertOffset(locate(CENTER, pose(roll=10), GimbalAngles(), CAMERA), 0, -20*math.tan(math.radians(10)))
        # Příď nahoru o 10° naklání kameru dopředu.
        self.assertOffset(locate(CENTER, pose(pitch=10), GimbalAngles(), CAMERA), 20*math.tan(math.radians(10)), 0)
        self.assertOffset(locate(CENTER, pose(pitch=10), GimbalAngles(forward=-10), CAMERA), 0, 0)

    def test_ray_above_horizon_has_no_ground_point(self):
        self.assertIsNone(locate(CENTER, pose(), GimbalAngles(forward=85), CAMERA))
        self.assertIsNone(locate(CENTER, pose(height=0), GimbalAngles(), CAMERA))

    def test_calibration_must_match_frame_size(self):
        with self.assertRaises(ValueError):
            locate(CENTER, pose(), GimbalAngles(), CAMERA, frame_size=(1920, 1080))

    def test_wgs84_degree_lengths(self):
        # Délka stupně na 50° šířky podle WGS84: 111 229 m (šířka), 71 696 m (délka).
        lat, lon = offset_to_latlon(50.0, 14.0, 111229.0, 0.0)
        self.assertAlmostEqual(lat, 51.0, delta=2e-4)
        lat, lon = offset_to_latlon(50.0, 14.0, 0.0, 71696.0)
        self.assertAlmostEqual(lon, 15.0, delta=2e-4)
        lat, lon = offset_to_latlon(50.0, 14.0, 10.0, 10.0)
        self.assertAlmostEqual((lat - 50.0) * 111229.0, 10.0, delta=0.01)
        self.assertAlmostEqual((lon - 14.0) * 71696.0, 10.0, delta=0.01)

    def test_distorted_lens_round_trip(self):
        # Soudkovité zkreslení jako u objektivu M12 3.6 mm; bod promítne OpenCV nezávisle.
        camera = CameraModel(np.array([[610.0, 0, 322], [0, 612.0, 236], [0, 0, 1]]),
                             np.array([-0.32, 0.12, 0.001, -0.0005, 0.0]), (640, 480))
        rng = np.random.default_rng(3)
        for _ in range(20):
            drone = pose(height=rng.uniform(5, 40), roll=rng.uniform(-8, 8), pitch=rng.uniform(-8, 8),
                         yaw=rng.uniform(0, 360))
            gimbal = GimbalAngles(rng.uniform(-40, 40), rng.uniform(-40, 40))
            rotation = body_in_ned(drone) @ camera_in_body(gimbal)
            target = rotation @ np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), 1.0])
            target = np.array([0, 0, -drone.height]) + target * drone.height / target[2]
            # Kamera ve výšce: bod v souřadnicích kamery = R^T (P - C).
            rvec, _ = cv2.Rodrigues(rotation.T)
            tvec = -rotation.T @ np.array([0, 0, -drone.height])
            pixel = cv2.projectPoints(target.reshape(1, 3), rvec, tvec, camera.matrix, camera.distortion)[0][0, 0]
            fix = locate(tuple(pixel), drone, gimbal, camera)
            self.assertAlmostEqual(fix.north, target[0], delta=0.01)
            self.assertAlmostEqual(fix.east, target[1], delta=0.01)


class AccuracyTests(unittest.TestCase):
    def test_error_grows_with_off_nadir_angle(self):
        errors = [locate(CENTER, pose(), GimbalAngles(forward=angle), CAMERA).error for angle in (0, 30, 60, 75)]
        self.assertEqual(errors, sorted(errors))
        self.assertGreater(errors[2], 2 * errors[0])
        # Kolmo dolů chyba výšky polohu nemění; zbývá hlavně GPS dronu a úhly.
        exact = Uncertainty(pixel=0, gimbal=0, attitude=0, yaw=0, height=5.0, gps=0)
        self.assertAlmostEqual(locate(CENTER, pose(), GimbalAngles(), CAMERA, uncertainty=exact).error, 0, delta=1e-6)
        self.assertAlmostEqual(locate(CENTER, pose(), GimbalAngles(forward=45), CAMERA, uncertainty=exact).error,
                               5.0, delta=0.01)

    def test_combine_prefers_overhead_measurement(self):
        oblique = locate(CENTER, pose(), GimbalAngles(forward=60), CAMERA)
        overhead = locate(CENTER, pose(lat=oblique.lat, lon=oblique.lon, height=15), GimbalAngles(), CAMERA)
        shifted = locate(CENTER, pose(lat=oblique.lat + 2e-5), GimbalAngles(forward=60), CAMERA)
        lat, lon, error = combine([shifted, overhead])
        self.assertLess(abs(lat - overhead.lat), abs(lat - shifted.lat))
        self.assertEqual(error, overhead.error)
        self.assertIsNone(combine([]))


class GimbalTests(unittest.TestCase):
    def test_servo_calibration_round_trip(self):
        axis = ServoAxis(sign=-1, scale=0.9, offset=2.5)
        self.assertAlmostEqual(axis.angle(axis.command(33.0)), 33.0)
        self.assertEqual(gimbal_from_servos(10, 20, ServoAxis(offset=1), ServoAxis(scale=0.5)), GimbalAngles(11, 10))

    def test_centering_moves_target_to_image_center(self):
        drone = pose(roll=3, pitch=-2, yaw=40)
        start = GimbalAngles(right=-10, forward=25)
        pixel = (505.0, 101.0)
        target = locate(pixel, drone, start, CAMERA)
        centered = centering_angles(pixel, start, CAMERA)
        principal = tuple(CAMERA.matrix[:2, 2])
        after = locate(principal, drone, centered, CAMERA)
        self.assertAlmostEqual(after.north, target.north, delta=0.01)
        self.assertAlmostEqual(after.east, target.east, delta=0.01)


class CalibrationTests(unittest.TestCase):
    def test_recovers_focal_length_from_rendered_chessboard(self):
        pattern, square_px, margin = (9, 6), 40, 60
        board = np.full((7*square_px + 2*margin, 10*square_px + 2*margin), 255, np.uint8)
        for row in range(7):
            for column in range(10):
                if (row + column) % 2 == 0:
                    y, x = margin + row*square_px, margin + column*square_px
                    board[y:y+square_px, x:x+square_px] = 0
        truth = np.array([[600.0, 0, 318], [0, 600.0, 243], [0, 0, 1]])
        square = 0.025
        # Rohy šachovnice (vnitřní roh 0) leží v bodě (0, 0) roviny desky.
        to_board = np.array([[square/square_px, 0, -(margin+square_px)*square/square_px],
                             [0, square/square_px, -(margin+square_px)*square/square_px], [0, 0, 1]])
        rng = np.random.default_rng(1)
        corner_sets = []
        for _ in range(100):
            if len(corner_sets) == 15:
                break
            rotation, _ = cv2.Rodrigues(np.radians(rng.uniform(-35, 35, 3)) * [1, 1, 0.3])
            translation = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.0), rng.uniform(0.35, 0.55)])
            homography = truth @ np.column_stack([rotation[:, 0], rotation[:, 1], translation]) @ to_board
            frame = cv2.warpPerspective(board, homography, (640, 480), flags=cv2.INTER_AREA,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=128)
            corners = find_corners(cv2.GaussianBlur(frame, (0, 0), 0.7), pattern)
            if corners is not None:
                corner_sets.append(corners)
        camera, rms = calibrate(corner_sets, pattern, square, (640, 480))
        self.assertLess(rms, 0.5)
        self.assertAlmostEqual(camera.matrix[0, 0], 600, delta=6)
        self.assertAlmostEqual(camera.matrix[1, 1], 600, delta=6)
        self.assertAlmostEqual(camera.matrix[0, 2], 318, delta=4)
        self.assertAlmostEqual(camera.matrix[1, 2], 243, delta=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'camera.json'
            camera.save(path, rms_px=rms)
            loaded = CameraModel.load(path)
        np.testing.assert_allclose(loaded.matrix, camera.matrix)
        self.assertEqual(loaded.size, (640, 480))

    def test_nominal_model_has_requested_field_of_view(self):
        self.assertAlmostEqual(CAMERA.fov[0], 54.0, places=6)
        self.assertAlmostEqual(CAMERA.fov[1], 42.0, places=6)


if __name__ == '__main__':
    unittest.main()
