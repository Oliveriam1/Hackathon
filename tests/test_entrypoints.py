"""Independent CLI/import boundaries and JSON handoff, with no real camera."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from src.models import Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, UAVTelemetry
from src.publisher import to_json
from src.vision import VisionProcessor

ROOT = Path(__file__).resolve().parent.parent


def run_program(name: str, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(ROOT / name), *args], cwd=ROOT,
                          input=input_text, text=True, capture_output=True, timeout=15)


class EntryPointTests(unittest.TestCase):
    def test_vision_dry_run_does_not_require_pi_library_or_camera(self) -> None:
        result = run_program("vision_main.py", "--dry-run", "--camera-backend", "picamera2", "--camera")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("camera access skipped", result.stderr)

    def test_navigation_dry_run_and_no_input(self) -> None:
        result = run_program("navigation_main.py", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "CENTERING")
        idle = run_program("navigation_main.py")
        self.assertEqual(json.loads(idle.stdout)["state"], "IDLE")

    def test_vision_json_is_navigation_input(self) -> None:
        frame = np.zeros((240, 320, 3), np.uint8)
        cv2.circle(frame, (80, 120), 20, (0, 255, 0), -1)
        vision = VisionProcessor().process_frame(frame)
        result = run_program("navigation_main.py", "--input", "-", input_text=to_json(vision) + "\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads(result.stdout)
        self.assertEqual(decision["state"], "CENTERING")
        self.assertAlmostEqual(decision["horizontal_error"], -0.5)

    def test_navigation_invalid_or_unavailable_input_is_idle(self) -> None:
        for value in ("garbage\n", "{}\n", "[]\n", ""):
            with self.subTest(value=value):
                result = run_program("navigation_main.py", "--input", "-", input_text=value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["state"], "IDLE")
        with tempfile.TemporaryDirectory() as directory:
            result = run_program("navigation_main.py", "--input", str(Path(directory) / "missing"))
            self.assertEqual(json.loads(result.stdout)["state"], "IDLE")

    def test_geolocation_dry_run_and_required_calibration(self) -> None:
        result = run_program("geolocation_main.py", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertTrue(data["synthetic"])
        self.assertAlmostEqual(data["location"]["north_offset_m"], 0)
        self.assertAlmostEqual(data["location"]["east_offset_m"], 0)
        missing = run_program("geolocation_main.py")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("CAMERA CALIBRATION REQUIRED", missing.stderr)
        self.assertEqual(missing.stdout, "")

    def test_red_candidate_pixel_can_be_geolocated_independently(self) -> None:
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(frame, (420, 240), 20, (0, 0, 255), -1)
        candidate = VisionProcessor().process_frame(frame).red_candidates[0]
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "camera.json"
            calibration.write_text(json.dumps({
                "intrinsics": asdict(CameraIntrinsics(200, 200, 320, 240, 640, 480)),
                "mount": asdict(CameraMount(0, -90, 0)),
            }))
            telemetry = Path(directory) / "telemetry.json"
            telemetry.write_text(to_json(UAVTelemetry(GeoCoordinate(50, 14), 10, Attitude(0, 0, 0))))
            result = run_program("geolocation_main.py", "--calibration", str(calibration),
                                 "--telemetry", str(telemetry), "--pixel",
                                 str(candidate.center.x), str(candidate.center.y))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(json.loads(result.stdout)["location"]["east_offset_m"], 5)
            missing = run_program("geolocation_main.py", "--calibration", str(calibration), "--pixel", "420", "240")
            self.assertEqual(missing.returncode, 1)
            self.assertIn("TELEMETRY REQUIRED", missing.stderr)

    def test_import_boundaries(self) -> None:
        cases = (
            (["navigation_main", "src.navigation"], ["cv2", "numpy", "src.camera", "src.detector", "src.geolocation"]),
            (["geolocation_main", "src.geolocation"], ["src.camera", "src.detector", "src.navigation", "src.field"]),
            (["vision_main", "src.vision"], ["src.geolocation", "src.navigation", "src.telemetry"]),
        )
        for imports, blocked in cases:
            code = f"""
import importlib
import sys
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {blocked!r} or fullname == 'picamera2':
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Blocker())
for name in {imports!r}:
    importlib.import_module(name)
"""
            with self.subTest(imports=imports):
                result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
