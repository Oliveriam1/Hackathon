"""Zpracování snímku a sestavení dat; nezávislé na GUI a zdroji obrazu."""
import time
from .vision import Vision
from .target_lock import TargetLock
from .publisher import detection_record
from .mission import Mission
from .drone_data import drone_record


def current_altitude(telemetry, fallback):
    """Označený zdroj odhadu; relative_alt není vzdálenost k terči."""
    if telemetry is not None:
        item = telemetry.snapshot()['messages'].get('GLOBAL_POSITION_INT')
        if item and item['age_s'] <= 0.5 and item['values']['relative_alt_m'] > 0.5:
            return {'value_m': item['values']['relative_alt_m'], 'source': 'telemetry_relative'}
    return {'value_m': fallback, 'source': 'manual' if fallback is not None else 'unknown'}


class DetectionPipeline:
    def __init__(self, config, *, telemetry=None, gimbal=None):
        self.telemetry = telemetry
        self.vision = Vision(mode=config.detector, expected_diameter=config.red_diameter_px)
        detector = self.vision.detector
        if config.detector == 'red':
            detector.redness_min, detector.r_min = config.redness_min, config.r_min
            detector.red_fraction_min = config.red_fraction
            detector.target_diameter_m, detector.hfov_deg = config.target_diameter_m, config.hfov_deg
            detector.analyze_geometry = config.geometry_debug
            if config.camera_calibration is not None:
                from .geolocation import CameraModel
                detector.camera_model = CameraModel.load(config.camera_calibration)
            detector.altitude_source = lambda: current_altitude(telemetry, config.altitude)
            if gimbal is not None:
                detector.gimbal_source = lambda: (gimbal.x, gimbal.y)
        detector.use_hough = config.hough
        detector.sensitivity = config.sensitivity
        self.target_lock = TargetLock()
        self.mission = Mission()
        self.sequence = 0

    def process(self, frame, *, received_at, sample_time, source):
        observation = self.vision.observe(frame, sample_time=sample_time)
        self.sequence += 1
        record = detection_record(observation, sequence=self.sequence,
                                  received_at=received_at, source=source)
        record['telemetry'] = self.telemetry.snapshot() if self.telemetry is not None else None
        record['visual_lock'] = self.target_lock.update(observation, sample_time=sample_time,
                                                      now=time.monotonic())
        record['autonomy'] = self.mission.snapshot()
        record['drone_data'] = drone_record(record)
        return observation, record

