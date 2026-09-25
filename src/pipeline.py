"""Zpracování snímku a sestavení dat; nezávislé na GUI a zdroji obrazu."""
import time
import math
from uuid import uuid4
from .vision import Vision
from .target_lock import TargetLock
from .publisher import detection_record
from .mission import Mission
from .drone_data import drone_record
from .target_locator import TargetLocator


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
        self.session_id = str(uuid4())
        self.config, self.gimbal = config, gimbal
        self.gimbal_controller = None
        self.gimbal_angles = None
        self.locator = TargetLocator(config)
        if config.track_camera and not config.gimbal_dry_run and gimbal is None:
            raise RuntimeError('Automatické sledování kamery vyžaduje připojená serva.')

    def follow_camera(self, observation, sample_time, source):
        if not self.config.track_camera:
            return {'enabled': False, 'state': 'DISABLED'}
        from .gimbal_controller import GimbalController
        from .geolocation import CameraModel, GimbalAngles
        if self.gimbal_controller is None:
            width, height = observation.frame_size
            if self.config.camera_calibration:
                model = CameraModel.load(self.config.camera_calibration)
            else:
                hfov = self.config.hfov_deg
                vfov = math.degrees(2*math.atan(height/width*math.tan(math.radians(hfov/2))))
                model = CameraModel.from_fov((width, height), (hfov, vfov))
            self.gimbal_controller = GimbalController(model, image_top=self.config.image_top,
                                                     max_speed=self.config.gimbal_speed)
            self.gimbal_angles = GimbalAngles()
        current = GimbalAngles(self.gimbal.x, self.gimbal.y) if self.gimbal else self.gimbal_angles
        command = None
        if source == 'camera':
            command = self.gimbal_controller.update(observation, current, sample_time=sample_time,
                                                    now=time.monotonic())
        else:
            self.gimbal_controller.status = 'NON_LIVE_SOURCE'
        if command is not None:
            if not self.config.gimbal_dry_run:
                self.gimbal.move_to(command.right, command.forward)
            self.gimbal_angles = command
        angles = command or current
        return dict(enabled=True, state=self.gimbal_controller.status,
                    dry_run=self.config.gimbal_dry_run, command_sent=command is not None and not self.config.gimbal_dry_run,
                    commanded_angles_deg={'right': angles.right, 'forward': angles.forward},
                    angle_basis='command_estimate_no_feedback')

    def process(self, frame, *, received_at, sample_time, source):
        observation = self.vision.observe(frame, sample_time=sample_time)
        self.sequence += 1
        record = detection_record(observation, sequence=self.sequence,
                                  received_at=received_at, source=source)
        record['telemetry'] = self.telemetry.snapshot() if self.telemetry is not None else None
        record['visual_lock'] = self.target_lock.update(observation, sample_time=sample_time,
                                                      now=time.monotonic())
        record['autonomy'] = self.mission.snapshot()
        # Úhly platné před novým povelem, nikoli poloha požadovaná až pro další snímek.
        from .geolocation import GimbalAngles
        angles, basis = None, 'unknown'
        if self.gimbal is not None and not self.config.gimbal_dry_run:
            angles = GimbalAngles(self.gimbal.x, self.gimbal.y)
            basis = 'command_estimate_no_feedback'
        elif self.config.camera_right_deg is not None and self.config.camera_forward_deg is not None:
            angles = GimbalAngles(self.config.camera_right_deg, self.config.camera_forward_deg)
            basis = 'user_fixed_mount'
        record['geolocation'], record['geolocation_status'] = self.locator.estimate(record, angles, basis)
        record['gimbal'] = self.follow_camera(observation, sample_time, source)
        record['drone_data'] = drone_record(record, session_id=self.session_id,
                                          max_age_s=self.target_lock.max_age_s)
        return observation, record

