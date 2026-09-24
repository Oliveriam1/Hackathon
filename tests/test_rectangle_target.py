import unittest
import cv2
import numpy as np
from src.vision import Vision


def scene(offset=0, rectangle=True, circle=True):
    frame = np.zeros((300, 500, 3), dtype=np.uint8)
    if rectangle:
        cv2.rectangle(frame, (80+offset, 50), (260+offset, 240), (230, 230, 230), 3)
    if circle:
        cv2.circle(frame, (170+offset, 145), 30, (180, 180, 180), -1)
    return frame


class TargetTests(unittest.TestCase):
    def test_faint_border_with_corner_gaps(self):
        frame = np.full((400, 400, 3), 60, dtype=np.uint8)
        for start, end in [((75, 65), (325, 65)), ((335, 75), (335, 325)),
                           ((325, 335), (75, 335)), ((65, 325), (65, 75))]:
            cv2.line(frame, start, end, (45, 45, 45), 2)
        cv2.circle(frame, (200, 200), 30, (15, 15, 15), -1)
        result = Vision().observe(frame)
        self.assertEqual(len(result.circles), 1)

    def test_square_and_skewed_quadrilateral(self):
        for corners in ([[80, 50], [270, 50], [270, 240], [80, 240]],
                        [[50, 50], [240, 50], [350, 240], [160, 240]]):
            with self.subTest(corners=corners):
                frame = np.zeros((300, 500, 3), dtype=np.uint8)
                polygon = np.array(corners, dtype=np.int32)
                cv2.polylines(frame, [polygon], True, (230, 230, 230), 3)
                center = tuple(polygon.mean(axis=0).astype(int))
                cv2.circle(frame, center, 25, (180, 180, 180), -1)
                observation = Vision().observe(frame)
                self.assertEqual(len(observation.circles), 1)
                self.assertAlmostEqual(observation.circles[0].x, center[0], delta=3)
                self.assertAlmostEqual(observation.circles[0].y, center[1], delta=3)

    def test_no_quad_skips_circle_search(self):
        from unittest.mock import patch
        vision = Vision()
        with patch.object(vision.detector, 'detect_prepared') as search:
            vision.observe(scene(rectangle=False))
            search.assert_not_called()

    def test_moving_target_confirmation_and_loss(self):
        vision = Vision()
        for index in range(4):
            observation = vision.observe(scene(index*4))
            self.assertEqual(len(observation.circles), 1)
            self.assertEqual(observation.confirmed, index == 3)
        # Krátký výpadek potvrzení neruší: cíl drží predikce (viz test_perspective_tracking).
        missing = vision.observe(scene(circle=False))
        self.assertTrue(missing.confirmed)
        self.assertFalse(missing.measured)
        self.assertEqual(missing.circles, [])

    def test_outside_circle_and_empty_rectangle(self):
        self.assertEqual(Vision().observe(scene(rectangle=False)).circles, [])
        self.assertEqual(Vision().observe(scene(circle=False)).circles, [])
        frame = scene(circle=False)
        cv2.circle(frame, (380, 145), 30, (255, 255, 255), -1)
        self.assertEqual(Vision().observe(frame).circles, [])

    def test_ambiguity_resets_confirmation(self):
        vision = Vision()
        frame = scene()
        cv2.rectangle(frame, (310, 50), (470, 240), (230, 230, 230), 3)
        cv2.circle(frame, (390, 145), 30, (180, 180, 180), -1)
        for _ in range(5):
            result = vision.observe(frame)
            self.assertFalse(result.confirmed)
            self.assertEqual(len(result.circles), 2)

    def test_setting_change_restarts_confirmation(self):
        vision = Vision()
        for _ in range(4):
            result = vision.observe(scene())
        self.assertTrue(result.confirmed)
        vision.detector.sensitivity = 2
        self.assertFalse(vision.observe(scene()).confirmed)
