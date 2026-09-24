import unittest

import cv2
import numpy as np

from src.target_detector import CircleInQuadrilateralDetector


class TargetDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = CircleInQuadrilateralDetector()

    @staticmethod
    def scene(circle=True, perspective=False):
        image = np.full((600, 900, 3), 110, dtype=np.uint8)
        if perspective:
            quad = np.array([[170, 100], [760, 135], [710, 520], [120, 470]], np.int32)
            center = (430, 305)
        else:
            quad = np.array([[150, 100], [750, 100], [750, 500], [150, 500]], np.int32)
            center = (450, 300)
        cv2.polylines(image, [quad], True, (225, 225, 225), 8)
        if circle:
            cv2.circle(image, center, 65, (25, 25, 25), 8)
        return image, center

    def test_circle_inside_rectangle(self):
        image, expected = self.scene(circle=True, perspective=False)
        result = self.detector.detect(image)
        self.assertTrue(result.targets)
        target = min(result.targets, key=lambda t: abs(t.center.x-expected[0]) + abs(t.center.y-expected[1]))
        self.assertLess(abs(target.center.x - expected[0]), 15)
        self.assertLess(abs(target.center.y - expected[1]), 15)

    def test_circle_inside_perspective_quadrilateral(self):
        image, expected = self.scene(circle=True, perspective=True)
        result = self.detector.detect(image)
        self.assertTrue(result.targets)
        target = min(result.targets, key=lambda t: abs(t.center.x-expected[0]) + abs(t.center.y-expected[1]))
        self.assertLess(abs(target.center.x - expected[0]), 25)
        self.assertLess(abs(target.center.y - expected[1]), 25)

    def test_quadrilateral_without_circle_is_not_target(self):
        image, _ = self.scene(circle=False)
        result = self.detector.detect(image)
        self.assertEqual(result.targets, ())

    def test_colour_cast_does_not_break_geometry(self):
        image, expected = self.scene(circle=True)
        # Simulate a severe magenta/red cast from an IR-sensitive camera.
        cast = image.astype(np.int16)
        cast[..., 0] = np.clip(cast[..., 0] + 45, 0, 255)
        cast[..., 2] = np.clip(cast[..., 2] + 75, 0, 255)
        cast[..., 1] = np.clip(cast[..., 1] - 35, 0, 255)
        result = self.detector.detect(cast.astype(np.uint8))
        self.assertTrue(result.targets)
        target = min(result.targets, key=lambda t: abs(t.center.x-expected[0]) + abs(t.center.y-expected[1]))
        self.assertLess(abs(target.center.x - expected[0]), 20)
        self.assertLess(abs(target.center.y - expected[1]), 20)


if __name__ == "__main__":
    unittest.main()
