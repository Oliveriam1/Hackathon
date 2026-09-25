"""Zpracování snímku a sestavení dat; nezávislé na GUI a zdroji obrazu."""
import time
from .vision import Vision
from .target_lock import TargetLock
from .publisher import detection_record
from .mission import Mission


def current_altitude(telemetry, fallback):
    """relative_alt z ArduPilotu, pokud je čerstvá a kladná, jinak --altitude."""
    if telemetry is not None:
        item = telemetry.snapshot()['messages'].get('GLOBAL_POSITION_INT')
        if item and item['age_s'] <= 0.5 and item['values']['relative_alt_m'] > 0.5:
            return item['values']['relative_alt_m']
    return fallback


class DetectionPipeline:
    def __init__(self, config, *, telemetry=None, gimbal=None):
        self.telemetry = telemetry
        self.vision = Vision(mode=config.detector, expected_diameter=config.red_diameter_px)
        detector = self.vision.detector
        if config.detector == 'red':
            detector.redness_min, detector.r_min = config.redness_min, config.r_min
            detector.red_fraction_min = config.red_fraction
            detector.altitude_source = lambda: current_altitude(telemetry, config.altitude)
            if gimbal is not None:
                detector.gimbal_source = lambda: (gimbal.x, gimbal.y)
        detector.use_hough = config.hough
        detector.sensitivity = config.sensitivity
        self.target_lock = TargetLock()
        self.mission = Mission()
        self.sequence = 0

    def process(self, frame, *, received_at, sample_time, source):
        observation = self.vision.observe(frame)
        self.sequence += 1
        record = detection_record(observation, sequence=self.sequence,
                                  received_at=received_at, source=source)
        record['telemetry'] = self.telemetry.snapshot() if self.telemetry is not None else None
        record['visual_lock'] = self.target_lock.update(observation, sample_time=sample_time,
                                                      now=time.monotonic())
        record['autonomy'] = self.mission.snapshot()
        return observation, record

