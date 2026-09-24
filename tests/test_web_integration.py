"""Náš Vision ve webovém streamu a v misi z větve test."""
import time
import unittest
from pathlib import Path

import numpy as np

from main import make_demo
from src.config import load_mission_config
from src.live_camera import StreamingDetector
from src.mission import MissionController
from src.models import MissionState
from src.simulation import SimulatedFlightController, StaticImageCamera
from src.streaming import LiveVideo, VideoSnapshot, draw_overlay
from src.vision import Vision

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'mission.example.json'


class RecordingPublisher:
    def __init__(self):
        self.sent = []

    def publish(self, coordinates):
        self.sent.append(coordinates)


class WebIntegrationTests(unittest.TestCase):
    def test_mission_reports_only_confirmed_target(self):
        config = load_mission_config(CONFIG)
        publisher = RecordingPublisher()
        report = MissionController(config, SimulatedFlightController(config.start), StaticImageCamera(make_demo()),
                                   Vision(), publisher).run()
        self.assertEqual(report.state, MissionState.COMPLETE)
        self.assertEqual(len(publisher.sent), 1)
        # Stejný výsledek jako detektor větve test v jejím README (x 10.3, y 13.4).
        self.assertAlmostEqual(publisher.sent[0].x, 10.3, delta=0.2)
        self.assertAlmostEqual(publisher.sent[0].y, 13.4, delta=0.2)

    def test_single_unconfirmed_frame_is_not_a_target(self):
        self.assertIsNone(Vision().detect_circle(make_demo()))

    def test_stream_overlay_draws_vision_observation(self):
        video = LiveVideo()
        camera = type('Camera', (), {'captured_at': time.monotonic()})()
        detector = StreamingDetector(Vision(), camera, video)
        frame = make_demo()
        for _ in range(4):
            observation = detector.detect(frame)
        self.assertTrue(observation.confirmed)
        self.assertIsNotNone(detector.detect_circle(frame))
        snapshot = VideoSnapshot(frame, 1, time.monotonic(), MissionState.SCANNING, None, 0,
                                 video.snapshot().detection, camera.captured_at, 'LIVE')
        output = draw_overlay(snapshot)
        self.assertFalse(np.array_equal(output[370:390, 610:630], frame[370:390, 610:630]))  # značka cíle


if __name__ == '__main__':
    unittest.main()
