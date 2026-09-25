"""Propojení kamery, mapy pásky, řadiče mise a letového backendu.

Kamera běží v hlavním vlákně (každý snímek -> pipeline, promítnutí pásky,
odhad cíle). Mise běží ve vlastním vlákně na 20 Hz, aby pomalé zpracování
obrazu nezpůsobilo mezeru v řízení (řadič vyžaduje krok <= 0.3 s).
Mezi vlákny se předává jen poslední hotový snímek pod zámkem.
"""
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from .approach import TargetEstimate
from .field import FieldMap, GroundPoint
from .geolocation import CameraModel, DronePose, GimbalAngles, _ground_offset
from .tape_mapper import TapeMapper, project_segments

MAX_TARGET_AGE = .25


@dataclass(frozen=True)
class VisionSample:
    processed_at: float
    sample_time: float
    target: TargetEstimate | None
    image_centered: bool
    state: str
    tape_segments: int
    green: TargetEstimate | None = None


def camera_model(config, size):
    if config.camera_calibration:
        return CameraModel.load(config.camera_calibration)
    width, height = size
    vfov = math.degrees(2*math.atan(height/width*math.tan(math.radians(config.hfov_deg/2))))
    return CameraModel.from_fov(tuple(size), (config.hfov_deg, vfov))


def target_from_record(record, vehicle, sample_time, now):
    """Cíl v lokální mapě = poloha dronu při snímku + posun z kamery.

    Sčítá se relativní posun, takže chyba GPS dronu se v navádění neprojeví
    (dron i tečka jsou ve stejné soustavě EKF). Vyžaduje čerstvé platné měření.
    """
    data = record.get('drone_data') or {}
    geo = record.get('geolocation')
    if (vehicle is None or not data.get('live_control_input_valid') or data.get('target_id') is None
            or record.get('geolocation_status') != 'ESTIMATED' or not geo):
        return None
    values = (geo.get('offset_north_m'), geo.get('offset_east_m'), geo.get('horizontal_error_estimate_m'),
              sample_time, now, vehicle.position.north_m, vehicle.position.east_m)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        return None
    if not 0 <= now-sample_time <= MAX_TARGET_AGE or not 0 <= values[2] <= 2:
        return None
    point = GroundPoint(vehicle.position.north_m+values[0], vehicle.position.east_m+values[1])
    return TargetEstimate(point, sample_time, str(data['target_id']), values[2])


class FlightRunner:
    def __init__(self, backend, controller, bounds, *, pipeline=None, config=None, gimbal=None,
                 log_stream=None, fake_camera=False, clock=time.monotonic,
                 green_detector=None, green_diameter_m=.2, green_gate_m=3.):
        self.backend, self.controller, self.pipeline = backend, controller, pipeline
        self.config, self.gimbal, self.clock = config, gimbal, clock
        self.log_stream, self.fake_camera = log_stream, fake_camera
        n0, n1, e0, e1 = bounds
        self.boundary = tuple(GroundPoint(*p) for p in ((n0, e0), (n1, e0), (n1, e1), (n0, e1)))
        self.mapper = TapeMapper()
        self.green_detector, self.green_diameter_m, self.green_gate_m = green_detector, green_diameter_m, green_gate_m
        self.green_candidates = 0
        self._lock = threading.Lock()
        self.vision = None
        self.model = None
        self.stop_requested = False
        self.last_command = None
        self.ticks = 0

    # ------------------------------------------------------------ kamera
    def camera_angles(self):
        if self.gimbal is not None:
            return GimbalAngles(self.gimbal.x, self.gimbal.y)
        if self.config.camera_right_deg is not None:
            return GimbalAngles(self.config.camera_right_deg, self.config.camera_forward_deg)
        return None

    def camera_step(self, frame, *, received_at, sample_time, source):
        vehicle, height, _ = self.backend.vehicle()
        angles = self.camera_angles()
        _, record = self.pipeline.process(frame, received_at=received_at, sample_time=sample_time, source=source)
        now = self.clock()
        target = target_from_record(record, vehicle, sample_time, now) if source == 'camera' else None
        tape = record.get('boundary') or {}
        count = 0
        green = None
        pose = None
        if source == 'camera' and vehicle is not None and angles is not None and math.isfinite(height) and height >= 1.:
            attitude = (record.get('telemetry') or {}).get('messages', {}).get('ATTITUDE')
            if attitude and attitude['age_s'] <= .2:
                a = attitude['values']
                if self.model is None:
                    self.model = camera_model(self.config, record['frame_size_px'])
                pose = DronePose(0., 0., height, a['roll_deg'], a['pitch_deg'], a['yaw_deg'])
        if pose is not None and tape.get('state') == 'TAPE_CANDIDATES' and tape.get('live_observation'):
            ground = project_segments(tape['segments_px'], pose, angles, self.model,
                                      vehicle.position, image_top=self.config.image_top)
            with self._lock:
                self.mapper.update(ground, sample_time)
            count = len(ground)
        if pose is not None and self.green_detector is not None:
            green = self.locate_green(frame, pose, angles, vehicle, sample_time)
        data = record['drone_data']
        with self._lock:
            self.vision = VisionSample(now, sample_time, target, bool(data.get('image_centered')),
                                       data.get('state'), count, green)
        return record

    def locate_green(self, frame, pose, angles, vehicle, sample_time):
        """Zelená tečka v lokální soustavě; vybere kandidáta nejblíž očekávané poloze."""
        expected_px = self.green_diameter_m*self.model.matrix[0, 0]/pose.height
        candidates = self.green_detector.detect(frame, expected_px)
        self.green_candidates = len(candidates)
        expected = self.controller.green_point or GroundPoint(0., 0.)
        best = None
        for c in candidates:
            offset = _ground_offset((c.x, c.y), pose, angles, self.model, self.config.image_top)
            if offset is None:
                continue
            point = GroundPoint(vehicle.position.north_m+offset[0], vehicle.position.east_m+offset[1])
            distance = math.hypot(point.north_m-expected.north_m, point.east_m-expected.east_m)
            if distance <= self.green_gate_m and (best is None or distance < best[0]):
                best = (distance, point)
        return TargetEstimate(best[1], sample_time, 'green', .1) if best else None

    # ------------------------------------------------------------ mise
    def mission_step(self, now=None):
        now = self.clock() if now is None else now
        with self._lock:
            vision = self.vision
            lines = self.mapper.lines
        field = FieldMap(self.boundary, now, lines)
        if self.fake_camera:
            camera_ready, locked, target, green = True, False, None, None
        else:
            camera_ready = vision is not None and 0 <= now-vision.processed_at <= .5
            fresh = vision is not None and 0 <= now-vision.sample_time <= MAX_TARGET_AGE
            target = vision.target if fresh else None
            green = vision.green if fresh else None
            locked = bool(fresh and vision.image_centered)
        data = self.backend.mission_input(field, camera_ready=camera_ready, camera_locked=locked,
                                          target=target, green=green, stop_requested=self.stop_requested)
        command = self.controller.update(data, now)
        sent = self.backend.execute(command)
        if self.pipeline is not None:
            self.pipeline.camera_mode = 'TRACK' if command.camera_action == 'TRACK' else 'SEARCH'
        self.last_command = command
        self.ticks += 1
        if self.log_stream is not None:
            row = dict(time_s=now, command=asdict(command), sent=sent, mission=self.controller.snapshot(),
                       vehicle=dict(north_m=data.vehicle.position.north_m, east_m=data.vehicle.position.east_m,
                                    height_m=data.height_m, guided=data.guided, armed=data.armed),
                       target=None if target is None else dict(north_m=target.position.north_m,
                                                               east_m=target.position.east_m,
                                                               error_m=target.error_m, id=target.identity),
                       green=None if green is None else dict(north_m=green.position.north_m,
                                                             east_m=green.position.east_m),
                       camera_ready=camera_ready, camera_locked=locked,
                       tape_lines=[[l.a.north_m, l.a.east_m, l.b.north_m, l.b.east_m, l.hits] for l in lines])
            self.log_stream.write(json.dumps(row, default=float)+'\n')
        return command, sent

    def mission_loop(self, stop, period=.05):
        next_tick = self.clock()
        while not stop.is_set():
            try:
                self.mission_step()
            except Exception as error:
                # Chyba v řízení: požádat o nulovou rychlost a skončit, pilot přebírá.
                print(f'MISE: chyba {error!r}; posílám nulovou rychlost a končím.', flush=True)
                try:
                    from .mission_controller import MissionCommand
                    self.backend.execute(MissionCommand('FAILSAFE', 'EXCEPTION', action='BRAKE'))
                finally:
                    stop.set()
                return
            next_tick += period
            delay = next_tick-self.clock()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = self.clock()
