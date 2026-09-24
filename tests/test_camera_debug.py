"""Calibration, CLI and frame processing tests; all capture/GUI calls are mocked."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import main
from src.camera import OpenCVCamera, load_camera_calibration
from src.detector import ColorDetector
from src.geolocation import TargetGeolocator
from src.models import Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, UAVTelemetry
from src.telemetry import StaticTelemetryProvider


INTRINSICS = CameraIntrinsics(200, 200, 320, 240, 640, 480)
NADIR = CameraMount(0, -90, 0)
TELEMETRY = UAVTelemetry(GeoCoordinate(50, 14), 10, Attitude(0, 0, 0))


class CalibrationTests(unittest.TestCase):
    def test_loads_explicit_calibration_and_mount(self) -> None:
        data = {"intrinsics": asdict(INTRINSICS), "mount": asdict(NADIR)}
        data["intrinsics"]["distortion_coefficients"] = [0, 0, 0, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            intrinsics, mount = load_camera_calibration(path)
        self.assertEqual(mount, NADIR)
        self.assertEqual(intrinsics.distortion_coefficients, (0, 0, 0, 0, 0))
        self.assertEqual(intrinsics.fx, 200)

    def test_missing_invalid_and_incomplete_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with self.assertRaises(ValueError):
                load_camera_calibration(path)
            for content in ("broken JSON", "{}", "[]", '{"intrinsics": null}',
                            json.dumps({"intrinsics": asdict(INTRINSICS)})):
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_camera_calibration(path)


class DebugOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.frame, (80, 80), (120, 120), (0, 0, 255), -1)
        cv2.rectangle(self.frame, (470, 320), (530, 380), (0, 0, 255), -1)
        cv2.rectangle(self.frame, (300, 220), (340, 260), (0, 255, 0), -1)
        self.detector = ColorDetector()
        self.geolocator = TargetGeolocator(INTRINSICS, NADIR)

    def test_all_red_objects_are_geolocated_and_green_is_centered(self) -> None:
        original = self.frame.copy()
        with patch.object(self.geolocator, "geolocate", wraps=self.geolocator.geolocate) as locate:
            with patch("cv2.putText", wraps=cv2.putText) as draw_text:
                overlay = main.draw_debug_overlay(self.frame, self.detector, self.geolocator, TELEMETRY)
        self.assertEqual(locate.call_count, 2)
        self.assertEqual({(call.args[0].x, call.args[0].y) for call in locate.call_args_list},
                         {(100, 100), (500, 350)})
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("GREEN error=(+0,+0) centered=True", texts)
        self.assertTrue(any(text.startswith("lat=") for text in texts))
        np.testing.assert_array_equal(self.frame, original)
        self.assertFalse(np.shares_memory(overlay, self.frame))

    def test_missing_inputs_and_mismatched_resolution(self) -> None:
        for geolocator, telemetry, expected in (
            (None, TELEMETRY, "CAMERA CALIBRATION REQUIRED"),
            (self.geolocator, None, "GEOLOCATION: NO TELEMETRY"),
            (None, None, "GEOLOCATION: CAMERA NOT CALIBRATED"),
            (TargetGeolocator(CameraIntrinsics(100, 100, 160, 120, 320, 240), NADIR),
             TELEMETRY, "GEOLOCATION: CALIBRATION SIZE MISMATCH"),
        ):
            with self.subTest(expected=expected), patch("cv2.putText", wraps=cv2.putText) as text:
                main.draw_debug_overlay(self.frame, self.detector, geolocator, telemetry)
                self.assertIn(expected, [call.args[1] for call in text.call_args_list])
                self.assertFalse(any(call.args[1].startswith("lat=") for call in text.call_args_list))

    def test_bad_ray_does_not_prevent_other_targets(self) -> None:
        # Forward-facing camera: upper target is above horizon, lower below it.
        geolocator = TargetGeolocator(INTRINSICS, CameraMount(0, 0, 0))
        with patch("cv2.putText", wraps=cv2.putText) as text:
            main.draw_debug_overlay(self.frame, self.detector, geolocator, TELEMETRY)
        texts = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any(value.startswith("GEOLOCATION: UNAVAILABLE") for value in texts))
        self.assertTrue(any(value.startswith("lat=") for value in texts))

    def test_mock_data_is_visibly_labelled(self) -> None:
        with patch("cv2.putText", wraps=cv2.putText) as text:
            main.draw_debug_overlay(
                self.frame, self.detector, self.geolocator, TELEMETRY, mock_telemetry=True
            )
        self.assertIn("STATIC TEST TELEMETRY - NOT REAL TARGET COORDINATES",
                      [call.args[1] for call in text.call_args_list])


class CameraLifecycleTests(unittest.TestCase):
    def test_constructor_never_opens_hardware(self) -> None:
        with patch("cv2.VideoCapture") as capture:
            camera = OpenCVCamera()
            capture.assert_not_called()
            with self.assertRaises(RuntimeError):
                camera.read()

    def test_capture_released_on_read_failure(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaises(RuntimeError):
                with OpenCVCamera() as camera:
                    camera.read()
        capture.release.assert_called_once()

    def test_unavailable_camera_released(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=capture), self.assertRaises(RuntimeError):
            with OpenCVCamera():
                self.fail("Unavailable camera was opened")
        capture.release.assert_called_once()

    def test_debug_loop_quits_and_releases_resources_without_telemetry(self) -> None:
        capture = Mock()
        capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        provider = Mock()
        provider.get_telemetry.side_effect = RuntimeError("No sample")
        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch("cv2.imshow") as show,
            patch("cv2.waitKey", return_value=ord("q")),
            patch("cv2.destroyAllWindows") as close,
        ):
            main.run_camera(0, None, provider)
        capture.release.assert_called_once()
        show.assert_called_once()
        close.assert_called_once()


class EntryPointTests(unittest.TestCase):
    def test_dry_run_never_opens_camera(self) -> None:
        with patch.object(sys, "argv", ["main.py", "--dry-run", "--camera"]):
            with patch("main.run_camera") as camera:
                with self.assertLogs("main", level="INFO") as logs:
                    self.assertEqual(main.main(), 0)
        camera.assert_not_called()
        self.assertTrue(any("CAMERA CALIBRATION REQUIRED" in line for line in logs.output))

    def test_calibrated_mock_mode_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"intrinsics": asdict(INTRINSICS), "mount": asdict(NADIR)}))
            args = ["main.py", "--camera", "--calibration", str(path), "--mock-telemetry"]
            with patch.object(sys, "argv", args), patch("main.run_camera") as camera:
                with self.assertLogs("main", level="INFO"):
                    self.assertEqual(main.main(), 0)
        camera.assert_called_once()
        self.assertIsInstance(camera.call_args.args[1], TargetGeolocator)
        self.assertIsInstance(camera.call_args.args[2], StaticTelemetryProvider)
        self.assertTrue(camera.call_args.kwargs["mock_telemetry"])

    def test_invalid_calibration_returns_one_before_opening_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            with patch.object(sys, "argv", ["main.py", "--camera", "--calibration", str(path)]):
                with patch("main.run_camera") as camera, self.assertLogs("main", level="ERROR"):
                    self.assertEqual(main.main(), 1)
        camera.assert_not_called()

    def test_imports_do_not_open_hardware_or_import_grid(self) -> None:
        code = """
import sys
import cv2
from unittest.mock import patch
with patch('cv2.VideoCapture', side_effect=AssertionError('Hardware accessed')):
    import main
    import src.camera
    import src.geolocation
    import src.telemetry
assert 'src.field' not in sys.modules
assert 'picamera2' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
