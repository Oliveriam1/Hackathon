"""Real loopback MJPEG, a 30 FPS dummy camera, and independent mission progress."""

from http.client import HTTPConnection
from pathlib import Path
import socket
import threading
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from main import make_demo
from src.config import load_mission_config
from src.live_camera import BufferedCamera, StreamingDetector
from src.mission import MissionController
from src.models import Attitude, GeoCoordinate, MissionState, Point, UAVTelemetry
from src.simulation import SimulatedFlightController, StaticImageCamera
from src.streaming import LiveVideo, draw_overlay
from src.target_detector import CircleInQuadrilateralDetector, DetectionResult, Target

ROOT = Path(__file__).resolve().parent.parent


class DummyCamera:
    """CPU-only changing scene; BufferedCamera schedules acquisition at 30 FPS."""
    def __init__(self):
        self.count = 0
        self.closed = False
        # Entropy fills a non-reading client's socket buffer quickly, exercising
        # actual backpressure rather than only a connection with spare capacity.
        self.background = np.random.default_rng(42).integers(0, 120, (360, 640, 3), dtype=np.uint8)

    def open(self):
        pass

    def read(self):
        self.count += 1
        frame = self.background.copy()
        cv2.circle(frame, (20 + self.count % 600, 240), 15, (200, 200, 200), -1)
        return frame

    def close(self):
        self.closed = True


def read_jpeg(response):
    assert response.readline().strip() == b"--frame"
    headers = {}
    while line := response.readline().strip():
        key, value = line.decode("ascii").split(":", 1)
        headers[key.lower()] = value.strip()
    assert headers["content-type"] == "image/jpeg"
    data = response.read(int(headers["content-length"]))
    assert response.read(2) == b"\r\n"
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image, int(headers["x-sequence"])


def test_overlay_geometry_telemetry_and_stale_data(monkeypatch):
    video = LiveVideo()
    frame = np.zeros((480, 640, 3), np.uint8)
    video.publish_frame(frame)
    video.set_state(MissionState.ASCENDING)
    video.set_telemetry(UAVTelemetry(GeoCoordinate(50, 14), 12.5, Attitude(0, 0, 0)))
    target = Target(Point(320, 300), 40,
                    (Point(100, 150), Point(550, 150), Point(550, 450), Point(100, 450)), 0.9)
    now = time.monotonic()
    video.set_detection(DetectionResult((target,), 1), now)
    real_put_text, labels = cv2.putText, []

    def put_text(image, text, *args, **kwargs):
        labels.append(text)
        return real_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    output = draw_overlay(video.snapshot(), now=now)
    assert not frame.any()  # Neither raw image nor shared snapshot is annotated in place.
    assert not video.snapshot().frame.any()
    assert output[300, 320].any()  # Circle center.
    assert output[150, 100].any()  # Quadrilateral.
    assert any("ASCENDING" in label for label in labels)
    assert any("12.50 m" in label for label in labels)
    assert any("50.000000, 14.000000" in label for label in labels)
    stale = draw_overlay(video.snapshot(), now=now + 2)
    assert not stale[300, 320].any()  # Do not draw old geometry as a current detection.
    assert any("CAMERA: STALE" in label for label in labels)
    assert any("DETECTION: STALE" in label for label in labels)
    video.set_telemetry(None)
    draw_overlay(video.snapshot())
    assert any("NO TELEMETRY" in label for label in labels)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 121])
def test_invalid_fps_does_not_start_workers(fps):
    with pytest.raises(ValueError, match="FPS"):
        LiveVideo(fps=fps)


def test_camera_loss_unblocks_analysis_and_closes_source():
    video, source = LiveVideo(), DummyCamera()
    source.read = Mock(side_effect=RuntimeError("camera disconnected"))
    camera = BufferedCamera(source, source.read, video)
    with pytest.raises(RuntimeError, match="unavailable"):
        camera.open()
    assert source.closed
    assert video.snapshot().camera_status == "ERROR"
    camera.close()  # Idempotent cleanup.


def test_capture_is_independent_of_slow_analysis():
    video, source = LiveVideo(), DummyCamera()
    camera = BufferedCamera(source, source.read, video)
    entered, release = threading.Event(), threading.Event()
    delegate = Mock()

    def slow_detect(frame, *, largest_only):
        entered.set()
        assert release.wait(2)
        return DetectionResult((), 0)

    delegate.detect.side_effect = slow_detect
    detector = StreamingDetector(delegate, camera, video)
    camera.open()
    thread = threading.Thread(target=lambda: detector.detect(camera.read(1)))
    try:
        thread.start()
        assert entered.wait(1)
        sequence = video.snapshot().sequence
        time.sleep(0.3)
        assert video.snapshot().sequence >= sequence + 5
        assert video.snapshot().detection is None
    finally:
        release.set()
        thread.join(2)
        camera.close()
    assert not thread.is_alive()
    assert video.snapshot().detection == DetectionResult((), 0)


def test_read_deadline_and_midstream_camera_loss():
    video, source = LiveVideo(), DummyCamera()
    release = threading.Event()
    original_read = source.read

    def read_then_disconnect():
        if source.count:
            assert release.wait(2)
            raise RuntimeError("camera disconnected during capture")
        return original_read()

    camera = BufferedCamera(source, read_then_disconnect, video)
    camera.open()
    try:
        with pytest.raises(TimeoutError, match="camera frame"):
            camera.read(0.05)
        release.set()
        with pytest.raises(RuntimeError, match="Camera"):
            camera.read(1)
    finally:
        release.set()
        camera.close()
    assert video.snapshot().camera_status == "ERROR"
    assert source.closed


def test_mjpeg_30fps_and_slow_client_do_not_block_navigation():
    source = DummyCamera()
    ticks = []
    stop_navigation = threading.Event()
    with LiveVideo(port=0, fps=30) as video:
        camera = BufferedCamera(source, source.read, video, fps=30)
        camera.open()

        def navigation_loop():
            while not stop_navigation.is_set():
                video.set_state(MissionState.SCANNING)
                ticks.append(time.monotonic())
                stop_navigation.wait(0.005)

        navigation = threading.Thread(target=navigation_loop)
        connection = HTTPConnection(video.host, video.port, timeout=3)
        # A second viewer requests the stream and then never consumes it.
        slow = socket.socket()
        slow.settimeout(2)
        slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        try:
            navigation.start()
            slow.connect((video.host, video.port))
            slow.sendall(b"GET /video_feed HTTP/1.0\r\nHost: localhost\r\n\r\n")
            connection.request("GET", "/video_feed")
            response = connection.getresponse()
            assert response.status == 200
            assert response.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=frame"
            first_frame, sequence = read_jpeg(response)
            started, captured, received = time.monotonic(), source.count, 0
            while time.monotonic() - started < 1.0:
                frame, next_sequence = read_jpeg(response)
                assert next_sequence > sequence
                sequence = next_sequence
                received += 1
            elapsed = time.monotonic() - started
            capture_fps = (source.count - captured) / elapsed
            received_fps = received / elapsed
            print(f"Dummy capture: {capture_fps:.1f} FPS; MJPEG: {received_fps:.1f} FPS; navigation ticks: {len(ticks)}")
            assert 20 <= capture_fps <= 35  # Allow loaded CI hosts and timer jitter.
            assert received_fps >= 20
            assert len(ticks) >= 60
            assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.3
            assert first_frame.shape == frame.shape == (360, 640, 3)
            assert not np.array_equal(first_frame, frame)
        finally:
            stop_navigation.set()
            navigation.join(2)
            connection.close()
            slow.close()
            camera.close()
    assert source.closed
    assert not navigation.is_alive()


def test_stream_during_blocked_mission_and_graceful_shutdown():
    config = load_mission_config(ROOT / "config/mission.example.json")
    flight = SimulatedFlightController(config.start)
    source = StaticImageCamera(make_demo())
    executing, release = threading.Event(), threading.Event()
    original_execute = flight.execute

    def delayed_execute(command, timeout_s):
        executing.set()
        assert release.wait(timeout_s)
        original_execute(command, timeout_s)

    flight.execute = delayed_execute
    publisher, reports = Mock(), []
    with LiveVideo(port=0, get_telemetry=flight.get_telemetry) as video:
        camera = BufferedCamera(source, lambda: source.read(1), video)
        detector = StreamingDetector(CircleInQuadrilateralDetector(), camera, video)
        controller = MissionController(config, flight, camera, detector, publisher, on_state=video.set_state)
        assert video.snapshot().state == MissionState.INIT
        runner = threading.Thread(target=lambda: reports.append(controller.run()))
        connection = HTTPConnection(video.host, video.port, timeout=3)
        try:
            runner.start()
            assert executing.wait(3)
            assert video.snapshot().state == MissionState.MOVING_TO_B
            # Read multiple actual JPEGs while execute() is deliberately blocked.
            connection.request("GET", "/video_feed")
            response = connection.getresponse()
            _, sequence = read_jpeg(response)
            for _ in range(5):
                _, next_sequence = read_jpeg(response)
            assert next_sequence > sequence
            assert video.snapshot().telemetry is not None
            assert runner.is_alive()
        finally:
            release.set()
            runner.join(5)
            connection.close()
        assert not runner.is_alive()
        assert reports[0].state == MissionState.COMPLETE, reports[0].error
        assert video.snapshot().state == MissionState.COMPLETE
        assert video.snapshot().camera_status == "STOPPED"
        assert video.snapshot().detection.targets
        publisher.publish.assert_called_once()
        assert not source.opened
    # A fresh server can bind the same port immediately after cleanup.
    with LiveVideo(port=video.port):
        pass


def test_http_index_unknown_path_and_unavailable_telemetry():
    provider = Mock(side_effect=RuntimeError("no telemetry"))
    with LiveVideo(port=0, get_telemetry=provider) as video:
        connection = HTTPConnection(video.host, video.port, timeout=2)
        try:
            connection.request("GET", "/")
            response = connection.getresponse()
            assert response.status == 200
            assert b'/video_feed' in response.read()
            connection.request("GET", "/missing")
            response = connection.getresponse()
            assert response.status == 404
            response.read()
            assert video.snapshot().telemetry is None
        finally:
            connection.close()
