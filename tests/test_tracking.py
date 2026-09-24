"""Tracking geometry, damaged frames, worker recovery and actual MJPEG throughput."""

from http.client import HTTPConnection
import threading
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from main import make_demo
from src.live_camera import BufferedCamera, StreamingDetector
from src.models import Point
from src.streaming import LiveVideo
from src.target_detector import CircleInQuadrilateralDetector, DetectionResult, Target
from src.tracking import EMPTY, TrackingDetector, tracker_factory
from tests.test_streaming import DummyCamera, read_jpeg

TARGET = Target(Point(160, 160), 20,
                (Point(80, 80), Point(240, 80), Point(240, 240), Point(80, 240)), 0.9)
RESULT = DetectionResult((TARGET,), 1)
FRAME = np.zeros((320, 320, 3), np.uint8)


class FakeTracker:
    def __init__(self, dx=0, dy=0):
        self.dx, self.dy = dx, dy
        self.ok = True

    def init(self, frame, box):
        self.box = box
        return None

    def update(self, frame):
        x, y, width, height = self.box
        return self.ok, (x + self.dx, y + self.dy, width, height)


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for worker")


def test_track_circle_and_quad_independently_without_redetection():
    detector = Mock()
    detector.detect.return_value = RESULT
    circle, quad = FakeTracker(10, -5), FakeTracker(4, 2)
    tracking = TrackingDetector(detector, factory=Mock(side_effect=[circle, quad]))
    assert tracking.process(FRAME, 0) == RESULT
    result = tracking.process(FRAME, 0.04)
    assert tracking.mode == "TRACKING"
    assert result.targets[0].center == Point(170, 155)
    assert result.targets[0].quadrilateral[0] == Point(84, 82)
    assert result.targets[0].radius_px == 20
    detector.detect.assert_called_once()


def test_loss_clears_state_then_reacquires_and_periodically_revalidates():
    detector = Mock()
    detector.detect.return_value = RESULT
    trackers = []

    def factory():
        tracker = FakeTracker()
        trackers.append(tracker)
        return tracker

    tracking = TrackingDetector(detector, factory=factory)
    tracking.process(FRAME, 0)
    trackers[0].ok = False
    assert tracking.process(FRAME, 0.04) == EMPTY
    assert tracking.mode == "LOST"
    assert tracking.process(FRAME, 0.05) is None  # Retain heavy-loop rate limit.
    assert tracking.process(FRAME, 0.13) == RESULT
    assert detector.detect.call_count == 2
    tracking.process(FRAME, 2.2)
    assert detector.detect.call_count == 3


def test_unavailable_contrib_falls_back_to_limited_detection(monkeypatch):
    monkeypatch.setattr("src.tracking.tracker_factory", lambda: None)
    detector = Mock()
    detector.detect.return_value = RESULT
    tracking = TrackingDetector(detector, detection_fps=5)
    for tick in range(10):
        tracking.process(FRAME, tick / 100)
    detector.detect.assert_called_once()
    assert tracking.process(FRAME, 0.21) == RESULT
    assert detector.detect.call_count == 2


@pytest.mark.parametrize("bad_frame", [None, np.zeros((0, 0, 3), np.uint8),
                                      np.zeros((10, 10, 2), np.uint8), FRAME.astype(np.float32)])
def test_invalid_analysis_frame_is_not_passed_to_opencv(bad_frame):
    detector = Mock()
    tracking = TrackingDetector(detector, factory=FakeTracker)
    assert tracking.process(bad_frame, 0) == EMPTY
    detector.detect.assert_not_called()


@pytest.mark.parametrize("box", [(float("nan"), 0, 10, 10), (-1, 0, 10, 10), (0, 0, 0, 10),
                                 (0, 0, 9999, 10)])
def test_invalid_tracker_boxes_cannot_reach_overlay(box):
    detector = Mock()
    detector.detect.return_value = RESULT
    tracker = FakeTracker()
    tracking = TrackingDetector(detector, factory=lambda: tracker)
    tracking.process(FRAME, 0)
    tracker.update = lambda frame: (True, box)
    assert tracking.process(FRAME, 0.04) == EMPTY


def test_real_kcf_tracks_translated_scene():
    factory = tracker_factory()
    if factory is None:
        pytest.skip("System OpenCV lacks contrib; fallback is covered separately")
    detector = Mock(wraps=CircleInQuadrilateralDetector())
    tracking = TrackingDetector(detector, factory=factory)
    frame = make_demo()
    initial = tracking.process(frame, 0).targets[0]
    # KCF's first update trains its appearance model on the initial frame.
    tracking.process(frame, 0.03)
    for step in range(1, 5):
        shifted = cv2.warpAffine(frame, np.float32([[1, 0, step * 3], [0, 1, step * 2]]),
                                 (1280, 720), borderValue=(115, 115, 115))
        result = tracking.process(shifted, 0.03 + step * 0.04)
        assert tracking.mode == "TRACKING"
        target = result.targets[0]
        assert target.center.x == pytest.approx(initial.center.x + step * 3, abs=5)
        assert target.center.y == pytest.approx(initial.center.y + step * 2, abs=5)
        assert target.quadrilateral[0].x == pytest.approx(initial.quadrilateral[0].x + step * 3, abs=7)
    detector.detect.assert_called_once()


def test_camera_recovers_from_none_empty_and_cv_errors():
    video, source = LiveVideo(), DummyCamera()
    original = source.read
    bad = iter([None, np.zeros((0, 0, 3), np.uint8), cv2.error("broken frame")])

    def read():
        item = next(bad, "good")
        if isinstance(item, Exception):
            raise item
        return original() if isinstance(item, str) else item

    camera = BufferedCamera(source, read, video)
    try:
        camera.open()
        assert camera.read(1).shape == (360, 640, 3)
        assert video.snapshot().camera_status == "LIVE"
    finally:
        camera.close()
    assert source.closed


def test_permanently_empty_camera_has_bounded_failure():
    video, source = LiveVideo(), DummyCamera()
    camera = BufferedCamera(source, lambda: None, video, max_frame_gap_s=0.1)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="unavailable"):
        camera.open()
    assert time.monotonic() - started < 2
    assert video.snapshot().camera_status == "ERROR"
    assert source.closed


def test_analysis_and_tracker_cv_errors_recover_in_worker():
    video, source = LiveVideo(), DummyCamera()
    camera = BufferedCamera(source, source.read, video)
    detector = Mock()
    detector.detect.side_effect = [cv2.error("damaged image"), RESULT, RESULT, RESULT]
    worker = StreamingDetector(detector, camera, video)
    faulty = FakeTracker()
    faulty.update = Mock(side_effect=cv2.error("tracker lost"))
    worker.tracking.factory = Mock(side_effect=[faulty, FakeTracker(), FakeTracker(), FakeTracker()])
    try:
        camera.open()
        worker.start()
        wait_until(lambda: video.snapshot().detection_mode == "TRACKING")
        assert detector.detect.call_count == 3
        assert video.snapshot().detection.targets
    finally:
        worker.close()
        camera.close()
    assert not worker._thread.is_alive()


def test_async_mission_consumer_rejects_stale_or_wrong_resolution_results():
    video, source = LiveVideo(), DummyCamera()
    camera = BufferedCamera(source, source.read, video)
    worker = StreamingDetector(Mock(), camera, video)
    worker._thread = threading.Thread()  # Select async consumer without starting work.
    camera._sequence = 2
    video.publish_frame(FRAME)
    video.set_detection(RESULT, time.monotonic(), sequence=2, shape=(320, 320))
    assert worker.detect_circle(FRAME) == TARGET.center
    video.set_detection(RESULT, time.monotonic() - 2, sequence=2, shape=(320, 320))
    assert worker.detect_circle(FRAME) is None
    video.set_detection(RESULT, time.monotonic(), sequence=2, shape=(360, 640))
    assert worker.detect_circle(FRAME) is None


def test_mjpeg_fps_with_slow_detection_and_missing_frames():
    source = DummyCamera()
    original_read, reads = source.read, [0]

    def sometimes_missing():
        reads[0] += 1
        return None if reads[0] % 9 == 0 else original_read()

    block, entered, release = threading.Event(), threading.Event(), threading.Event()
    detector = Mock()
    calls = []

    def detect(frame, *, largest_only):
        calls.append((threading.current_thread().name, time.monotonic()))
        if block.is_set():
            entered.set()
            assert release.wait(3)
        return EMPTY

    detector.detect.side_effect = detect
    with LiveVideo(port=0, fps=30) as video:
        camera = BufferedCamera(source, sometimes_missing, video)
        worker = StreamingDetector(detector, camera, video, detection_fps=8)
        connection = HTTPConnection(video.host, video.port, timeout=3)
        try:
            camera.open()
            worker.start()
            connection.request("GET", "/video_feed")
            response = connection.getresponse()
            assert response.status == 200

            def measure_fps():
                started, count, sequences = time.monotonic(), 0, []
                while time.monotonic() - started < 1:
                    frame, sequence = read_jpeg(response)
                    assert frame.shape == (360, 640, 3)  # Native resolution.
                    sequences.append(sequence)
                    count += 1
                assert sequences == sorted(set(sequences))
                return count / (time.monotonic() - started)

            baseline = measure_fps()
            block.set()
            assert entered.wait(1)
            before = video.snapshot().sequence
            blocked = measure_fps()
            assert video.snapshot().sequence >= before + 20
            assert blocked >= 20 and blocked >= baseline * 0.8
            assert all(name == "vision-analysis" for name, _ in calls)
            assert all(b[1] - a[1] >= 0.12 for a, b in zip(calls, calls[1:]))
            print(f"MJPEG baseline={baseline:.1f} FPS, blocked analysis={blocked:.1f} FPS, with dropped frames")
        finally:
            release.set()
            worker.close()
            camera.close()
            connection.close()
