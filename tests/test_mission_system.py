"""Navigation, mapping, five static scenes, real loopback UDP and simulated E2E."""

from dataclasses import replace
import json
import math
from pathlib import Path
import socket
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from main import make_demo
from src.config import load_mission_config
from src.geolocation import geodetic_to_local, geolocate, local_to_geodetic
from src.mission import MissionController
from src.models import Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, LocalPoint, MissionState, Point, UAVTelemetry
from src.navigation import plan_mission, polygon_centroid, required_scan_altitude
from src.publisher import UDPPublisher
from src.pi_camera import PiCamera
from src.simulation import SimulatedFlightController, StaticImageCamera
from src.target_detector import CircleInQuadrilateralDetector

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).parent / "assets"
RECTANGLE = (LocalPoint(0, 0), LocalPoint(20, 0), LocalPoint(20, 10), LocalPoint(0, 10))


def test_centroid_square_and_asymmetric_quadrilateral():
    assert polygon_centroid(RECTANGLE) == LocalPoint(10, 5)
    trapezoid = (LocalPoint(0, 0), LocalPoint(4, 0), LocalPoint(2, 2), LocalPoint(0, 2))
    centroid = polygon_centroid(trapezoid)
    assert centroid.x == pytest.approx(14 / 9)
    assert centroid.y == pytest.approx(8 / 9)
    assert polygon_centroid(tuple(reversed(trapezoid))) == centroid


def test_coverage_height_and_yaw():
    assert required_scan_altitude(RECTANGLE, 90, 90, coverage_margin=1) == pytest.approx(10)
    assert required_scan_altitude(RECTANGLE, 90, 60, coverage_margin=1) == pytest.approx(10)
    assert required_scan_altitude(RECTANGLE, 90, 60, yaw_deg=90, coverage_margin=1) == pytest.approx(10 / math.tan(math.pi / 6))


def test_asymmetric_height_uses_farthest_corner():
    corners = (LocalPoint(0, 0), LocalPoint(4, 0), LocalPoint(2, 2), LocalPoint(0, 2))
    assert required_scan_altitude(corners, 90, 90, coverage_margin=1) == pytest.approx(22 / 9)


@pytest.mark.parametrize("fov", [0, 180, -1, float("nan")])
def test_invalid_fov(fov):
    with pytest.raises(ValueError):
        required_scan_altitude(RECTANGLE, fov, 60)


def test_invalid_corners_and_ceiling():
    with pytest.raises(ValueError):
        polygon_centroid(tuple(LocalPoint(i, i) for i in range(4)))
    config = load_mission_config(ROOT / "config/mission.example.json")
    with pytest.raises(ValueError, match="exceeds"):
        plan_mission(config.start, config.destination, config.corners, 70, 50, 5, 6)


@pytest.mark.parametrize("case", json.loads((ASSETS / "expected.json").read_text()), ids=lambda case: case["file"])
def test_five_static_images_with_five_pixel_tolerance(case):
    # imdecode: cv2.imread na Windows neotevře cestu s diakritikou/azbukou.
    frame = cv2.imdecode(np.fromfile(ASSETS / case["file"], np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    center = CircleInQuadrilateralDetector().detect_circle(frame)
    assert center is not None
    assert abs(center.x - case["center"][0]) <= 5
    assert abs(center.y - case["center"][1]) <= 5


def test_no_circle_returns_none():
    frame = np.full((600, 900, 3), 110, np.uint8)
    cv2.rectangle(frame, (150, 100), (750, 500), (230, 230, 230), 8)
    assert CircleInQuadrilateralDetector().detect_circle(frame) is None


@pytest.mark.parametrize("pixel,attitude,north,east", [
    (Point(320, 240), Attitude(0, 0, 0), 0, 0),
    (Point(420, 240), Attitude(0, 0, 0), 0, 5),
    (Point(320, 140), Attitude(0, 0, 0), 5, 0),
    (Point(420, 240), Attitude(0, 0, 90), -5, 0),
    (Point(320, 240), Attitude(0, 45, 0), 10, 0),
    (Point(320, 240), Attitude(45, 0, 0), 0, -10),
])
def test_pixel_to_ground(pixel, attitude, north, east):
    result = geolocate(pixel, UAVTelemetry(GeoCoordinate(50, 14), 10, attitude),
                        CameraIntrinsics(200, 200, 320, 240, 640, 480), CameraMount(0, -90, 0))
    assert result.north_offset_m == pytest.approx(north, abs=1e-10)
    assert result.east_offset_m == pytest.approx(east, abs=1e-10)


def test_invalid_ray_and_agl():
    k = CameraIntrinsics(200, 200, 320, 240, 640, 480)
    for height, mount in ((0, CameraMount(0, -90, 0)), (10, CameraMount(0, 0, 0)),
                          (10, CameraMount(0, 90, 0))):
        with pytest.raises(ValueError):
            geolocate(Point(320, 240), UAVTelemetry(GeoCoordinate(50, 14), height, Attitude(0, 0, 0)), k, mount)


def test_wgs84_short_distance_reference_and_roundtrip():
    origin = GeoCoordinate(0, 0)
    target = local_to_geodetic(origin, LocalPoint(10, 10))
    assert target.latitude == pytest.approx(0.000090436947705, abs=1e-12)
    assert target.longitude == pytest.approx(0.000089831528412, abs=1e-12)
    point = geodetic_to_local(origin, target)
    assert point.x == pytest.approx(10, abs=1e-8)
    assert point.y == pytest.approx(10, abs=1e-8)


def test_udp_and_full_mission_end_to_end():
    config = load_mission_config(ROOT / "config/mission.example.json")
    flight = SimulatedFlightController(config.start)
    camera = StaticImageCamera(make_demo())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
        server.bind(("127.0.0.1", 0))
        server.settimeout(2)
        publisher = UDPPublisher("127.0.0.1", server.getsockname()[1])
        report = MissionController(config, flight, camera, CircleInQuadrilateralDetector(), publisher).run()
        assert report.state == MissionState.COMPLETE, report.error
        data, _ = server.recvfrom(4096)
    payload = json.loads(data)
    assert payload == {"target_found": True, "coordinates": {"x": report.coordinates.x, "y": report.coordinates.y}}
    assert report.history == tuple(MissionState)[0:8]
    assert len(flight.commands) == 4
    assert not camera.opened
    assert not flight.aborted


def test_no_target_times_out_at_60_seconds_without_sending():
    config = load_mission_config(ROOT / "config/mission.example.json")
    flight = SimulatedFlightController(config.start)
    camera = Mock()
    clock = [0.0]
    def read(timeout_s):
        clock[0] += min(10.0, timeout_s)
        return make_demo()
    camera.read.side_effect = read
    detector, publisher = Mock(), Mock()
    detector.detect_circle.return_value = None
    report = MissionController(config, flight, camera, detector, publisher,
                               clock=lambda: clock[0], sleep=lambda seconds: None).run()
    assert report.state == MissionState.FAILED
    assert clock[0] == 60
    assert flight.aborted
    publisher.publish.assert_not_called()
    camera.close.assert_called_once()


@pytest.mark.parametrize("failure", ["camera", "telemetry", "command", "publisher"])
def test_failures_abort_and_close(failure):
    config = load_mission_config(ROOT / "config/mission.example.json")
    flight = Mock(wraps=SimulatedFlightController(config.start))
    camera = Mock(wraps=StaticImageCamera(make_demo()))
    publisher = Mock()
    if failure == "camera":
        camera.read.side_effect = RuntimeError("camera signal lost")
    elif failure == "telemetry":
        flight.get_telemetry.side_effect = RuntimeError("telemetry lost")
    elif failure == "command":
        flight.execute.side_effect = TimeoutError("command completion timeout")
    else:
        publisher.publish.side_effect = OSError("network unreachable")
    report = MissionController(config, flight, camera, CircleInQuadrilateralDetector(), publisher).run()
    assert report.state == MissionState.FAILED
    flight.abort.assert_called_once()
    camera.close.assert_called_once()
    if failure != "publisher":
        publisher.publish.assert_not_called()


def test_invalid_plan_sends_no_motion_commands():
    config = replace(load_mission_config(ROOT / "config/mission.example.json"), max_altitude_agl_m=6)
    flight, camera, publisher = Mock(), Mock(), Mock()
    report = MissionController(config, flight, camera, Mock(), publisher).run()
    assert report.state == MissionState.FAILED
    flight.execute.assert_not_called()
    camera.open.assert_not_called()


def test_pi_camera_enables_normal_color_capture_without_hardware(monkeypatch):
    device = Mock()
    device.capture_array.return_value = make_demo()
    factory = Mock(return_value=device)
    factory.global_camera_info.return_value = [{"Id": "test-camera"}]
    monkeypatch.setattr("src.pi_camera._load_picamera2_class", lambda: factory)
    monkeypatch.setattr("src.pi_camera.time.sleep", lambda seconds: None)
    camera = PiCamera()
    camera.open()
    settings = device.create_video_configuration.call_args.kwargs
    assert settings["main"]["format"] == "RGB888"
    assert settings["controls"] == {"AwbEnable": True, "AeEnable": True}
    assert CircleInQuadrilateralDetector().detect_circle(camera.read()) is not None
    camera.close()
    device.stop.assert_called_once()
    device.close.assert_called_once()


def test_unconfirmed_scan_pose_does_not_publish():
    config = load_mission_config(ROOT / "config/mission.example.json")
    flight = SimulatedFlightController(config.start)
    flight.execute = Mock()  # Claims completion but position/height never change.
    publisher = Mock()
    report = MissionController(config, flight, StaticImageCamera(make_demo()),
                               CircleInQuadrilateralDetector(), publisher).run()
    assert report.state == MissionState.FAILED
    assert "pose" in report.error
    publisher.publish.assert_not_called()
