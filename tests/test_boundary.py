import unittest
import cv2
import numpy as np
from src.boundary_detector import RedWhiteTapeDetector
from src.zone_guard import ZoneGuard
from src.field import FieldMap, GroundPoint
from src.approach import VehicleState, ApproachCommand
from src.config import parse_args
from src.pipeline import DetectionPipeline
import time


class BoundaryTests(unittest.TestCase):
    def tape(self):
        frame = np.full((240, 320, 3), (30, 80, 30), np.uint8)
        for i, x in enumerate(range(30, 291, 20)):
            cv2.rectangle(frame, (x, 100), (x+19, 114), (0, 0, 255) if i % 2 else (240, 240, 240), -1)
        return frame

    def test_alternating_tape_and_rotation(self):
        frame = self.tape()
        for angle in (0, 35, 90):
            rotated = cv2.warpAffine(frame, cv2.getRotationMatrix2D((160, 120), angle, .8), (320, 240))
            self.assertTrue(RedWhiteTapeDetector().detect(rotated, sample_time=1).segments_px, angle)

    def test_dot_solid_red_blank_not_tape(self):
        for kind in ('dot', 'line', 'blank'):
            frame = np.full((240, 320, 3), 240, np.uint8)
            if kind == 'dot': cv2.circle(frame, (150, 120), 20, (0, 0, 255), -1)
            if kind == 'line': cv2.rectangle(frame, (20, 100), (300, 115), (0, 0, 255), -1)
            self.assertFalse(RedWhiteTapeDetector().detect(frame, sample_time=1).segments_px)

    def test_pipeline_never_treats_segments_as_verified_zone(self):
        pipeline = DetectionPipeline(parse_args(['--detect-boundary', '--demo']))
        observation, record = pipeline.process(self.tape(), received_at=time.time(),
                                                sample_time=time.monotonic(), source='image')
        self.assertTrue(record['boundary']['segments_px'])
        self.assertFalse(record['drone_data']['boundary']['movement_allowed'])
        self.assertEqual(record['boundary']['zone_state'], 'ZONE_UNKNOWN')
        self.assertFalse(record['boundary']['live_observation'])

    def test_guard_unknown_stale_outside_braking(self):
        guard = ZoneGuard()
        corners = tuple(GroundPoint(*p) for p in ((-10, -10), (10, -10), (10, 10), (-10, 10)))
        field = FieldMap(corners, 1)
        def check(n=0, speed=0, field=field, now=1):
            return guard.check(field, VehicleState(GroundPoint(n, 0), speed, 0, now),
                               ApproachCommand('TEST', speed, 0), now)
        self.assertTrue(check().allowed)
        self.assertTrue(check(field=FieldMap(corners[::-1], 1)).allowed)
        self.assertEqual(check(field=FieldMap()).state, 'ZONE_UNKNOWN')
        self.assertEqual(check(now=4).state, 'ZONE_STALE')
        self.assertEqual(check(n=11).state, 'ZONE_OUTSIDE')
        self.assertEqual(check(n=8, speed=2).state, 'ZONE_BRAKE')
        self.assertTrue(check(n=8, speed=0).allowed)
        self.assertFalse(check(n=float('nan')).allowed)

    def test_concave_boundary_rejected(self):
        field = FieldMap(tuple(GroundPoint(*p) for p in ((0, 0), (5, 0), (2, 2), (5, 5), (0, 5))), 1)
        decision = ZoneGuard().check(field, VehicleState(GroundPoint(1, 1), 0, 0, 1),
                                     ApproachCommand('TEST'), 1)
        self.assertFalse(decision.allowed)
