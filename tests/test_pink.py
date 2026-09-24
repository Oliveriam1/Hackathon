"""Temporary pink detection tests; no real camera or GUI access."""

import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import main
from src.camera import run_camera_preview
from src.detector import ColorDetector
from src.models import Point


class PinkTests(unittest.TestCase):
    def test_regions_centers_noise_and_color_separation(self) -> None:
        hsv = np.zeros((240, 320, 3), dtype=np.uint8)
        hsv[20:61, 20:61] = (150, 180, 255)
        hsv[100:161, 200:261] = (165, 80, 255)
        hsv[180:183, 10:13] = (150, 255, 255)
        hsv[180:188, 30:38] = (150, 255, 255)
        hsv[180:211, 80:111] = (0, 255, 255)
        hsv[180:211, 130:161] = (175, 255, 255)
        hsv[180:211, 180:211] = (60, 255, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        detector = ColorDetector()
        detections = detector.detect_pink_test(frame)
        self.assertEqual([item.center for item in detections], [Point(230, 130), Point(40, 40)])
        self.assertEqual(detections[0].bounding_box, (200, 100, 61, 61))
        self.assertAlmostEqual(detections[0].normalized_center.x, 230 / 320)
        self.assertAlmostEqual(detections[0].normalized_center.y, 130 / 240)
        self.assertEqual(len(detector.detect_red(frame)), 2)
        self.assertIsNotNone(detector.detect_green(frame))

    def test_empty_and_invalid_frames(self) -> None:
        detector = ColorDetector()
        self.assertEqual(detector.detect_pink_test(np.zeros((100, 100, 3), np.uint8)), [])
        with self.assertRaises(ValueError):
            detector.detect_pink_test(np.zeros((100, 100), np.uint8))

    def test_pink_runs_only_when_enabled(self) -> None:
        for enabled in (False, True):
            capture = Mock()
            capture.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))
            with (
                self.subTest(enabled=enabled),
                patch("cv2.VideoCapture", return_value=capture),
                patch("cv2.imshow"),
                patch("cv2.waitKey", return_value=ord("q")),
                patch("cv2.destroyAllWindows"),
                patch.object(ColorDetector, "detect_pink_test", return_value=[]) as pink,
            ):
                run_camera_preview(test_pink=enabled)
            self.assertEqual(pink.call_count, int(enabled))
            capture.release.assert_called_once()

    def test_camera_released_if_frame_read_fails(self) -> None:
        capture = Mock()
        capture.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=capture), self.assertRaises(RuntimeError):
            run_camera_preview(test_pink=True)
        capture.release.assert_called_once()

    def test_dry_run_skips_camera_and_cli_passes_flag(self) -> None:
        with patch.object(sys, "argv", ["main.py", "--camera", "--test-pink", "--dry-run"]):
            with patch("src.camera.run_camera_preview") as preview:
                self.assertEqual(main.main(), 0)
                preview.assert_not_called()
        with patch.object(sys, "argv", ["main.py", "--camera", "--test-pink"]):
            with patch("src.camera.run_camera_preview") as preview:
                self.assertEqual(main.main(), 0)
                preview.assert_called_once_with(0, test_pink=True)


if __name__ == "__main__":
    unittest.main()
