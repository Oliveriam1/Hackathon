"""Synthetic Vision frames and mocked camera backends."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.camera import OpenCVCamera, Picamera2Camera
from src.models import Point
from src.publisher import ConsolePublisher
from src.vision import VisionProcessor, draw_vision_overlay, run_vision


class VisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.processor = VisionProcessor()

    def test_green_centered(self) -> None:
        cv2.circle(self.frame, (160, 120), 20, (0, 255, 0), -1)
        result = self.processor.process_frame(self.frame, timestamp=123.5)
        self.assertTrue(result.centering.centered)
        self.assertEqual(result.green_detection.center, Point(160, 120))
        self.assertEqual(result.timestamp, 123.5)

    def test_green_left(self) -> None:
        cv2.circle(self.frame, (80, 120), 20, (0, 255, 0), -1)
        result = self.processor.process_frame(self.frame)
        self.assertLess(result.centering.error_x, 0)
        self.assertAlmostEqual(result.centering.normalized_error_x, -0.5)
        self.assertFalse(result.centering.centered)

    def test_red_circle_is_candidate(self) -> None:
        cv2.circle(self.frame, (100, 100), 20, (0, 0, 255), -1)
        result = self.processor.process_frame(self.frame)
        self.assertEqual(len(result.red_candidates), 1)
        self.assertEqual(result.red_candidates[0].center, Point(100, 100))

    def test_blank_frame(self) -> None:
        result = self.processor.process_frame(self.frame)
        self.assertEqual(result.red_candidates, ())
        self.assertIsNone(result.green_detection)
        self.assertFalse(result.centering.detected)

    def test_boundary_tape_is_never_confirmed(self) -> None:
        for x in range(0, 300, 30):
            color = (0, 0, 255) if x % 60 == 0 else (255, 255, 255)
            cv2.rectangle(self.frame, (x, 20), (x + 29, 45), color, -1)
        result = self.processor.process_frame(self.frame)
        output = io.StringIO()
        ConsolePublisher(output).publish(result)
        data = json.loads(output.getvalue())
        self.assertGreater(len(data["red_candidates"]), 0)
        self.assertNotIn("confirmed", output.getvalue())
        self.assertNotIn("target", output.getvalue())
        self.assertIn("normalized_center", data["red_candidates"][0])

    def test_filter_extension_preserves_candidate_contract(self) -> None:
        cv2.circle(self.frame, (100, 100), 20, (0, 0, 255), -1)
        processor = VisionProcessor(candidate_filter=lambda frame, candidates: ())
        self.assertEqual(processor.process_frame(self.frame).red_candidates, ())

    def test_overlay_does_not_modify_frame(self) -> None:
        result = self.processor.process_frame(self.frame)
        overlay = draw_vision_overlay(self.frame, result)
        self.assertFalse(self.frame.any())
        self.assertTrue(overlay.any())
        self.assertFalse(np.shares_memory(overlay, self.frame))

    def test_headless_capture_publishes_monotonic_timestamps(self) -> None:
        camera = Mock()
        camera.read.return_value = self.frame
        publisher = Mock()
        with patch("src.vision.time.monotonic", side_effect=[10.0, 11.0]), patch("cv2.imshow") as show:
            run_vision(camera, self.processor, publisher, max_frames=2)
        self.assertEqual([call.args[0].timestamp for call in publisher.publish.call_args_list], [10, 11])
        show.assert_not_called()
        camera.close.assert_called_once()

    def test_publisher_failure_closes_camera(self) -> None:
        camera, publisher = Mock(), Mock()
        camera.read.return_value = self.frame
        publisher.publish.side_effect = RuntimeError("output failed")
        with self.assertRaises(RuntimeError):
            run_vision(camera, self.processor, publisher, max_frames=1)
        camera.close.assert_called_once()


class BackendTests(unittest.TestCase):
    def test_opencv_is_lazy_and_closes(self) -> None:
        capture = Mock()
        capture.read.return_value = (True, np.zeros((20, 20, 3), np.uint8))
        with patch("cv2.VideoCapture", return_value=capture) as factory:
            camera = OpenCVCamera()
            factory.assert_not_called()
            with self.assertRaises(RuntimeError):
                camera.read()
            camera.start()
            self.assertEqual(camera.read().shape, (20, 20, 3))
            camera.close()
            camera.close()
        capture.release.assert_called_once()

    def test_unavailable_opencv_camera_is_released(self) -> None:
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=capture), self.assertRaises(RuntimeError):
            OpenCVCamera().start()
        capture.release.assert_called_once()

    def test_picamera_unavailable_does_not_break_import_or_constructor(self) -> None:
        with patch.dict("sys.modules", {"picamera2": None}):
            camera = Picamera2Camera()
            with self.assertRaisesRegex(RuntimeError, "Picamera2 is not available"):
                camera.start()
            camera.close()

    def test_picamera_rgb888_is_returned_as_bgr(self) -> None:
        camera, factory = Mock(), Mock()
        factory.return_value = camera
        # Picamera2 RGB888 memory order is BGR: this is a red image.
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[80:121, 80:121] = (0, 0, 255)
        camera.capture_array.return_value = frame
        with patch.dict("sys.modules", {"picamera2": SimpleNamespace(Picamera2=factory)}):
            backend = Picamera2Camera(width=320, height=240)
            factory.assert_not_called()
            backend.start()
            result = VisionProcessor().process_frame(backend.read())
            self.assertEqual(len(result.red_candidates), 1)
            camera.create_preview_configuration.assert_called_once_with(
                main={"size": (320, 240), "format": "RGB888"}
            )
            backend.close()
            backend.close()
        camera.stop.assert_called_once()
        camera.close.assert_called_once()

    def test_picamera_setup_failure_is_closed(self) -> None:
        camera = Mock()
        camera.configure.side_effect = RuntimeError("configuration failed")
        with patch.dict("sys.modules", {"picamera2": SimpleNamespace(Picamera2=Mock(return_value=camera))}):
            with self.assertRaises(RuntimeError):
                Picamera2Camera().start()
        camera.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
