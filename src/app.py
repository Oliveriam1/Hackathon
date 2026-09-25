"""Hlavní smyčka: kamera -> detekce -> míření serv -> úhel (-> souřadnice)."""
import json
import math
import os
import sys
import time
from contextlib import ExitStack
import cv2
from .aim import Aimer, AngleAverager
from .geometry import CameraModel, GimbalAngles
from .locate import DronePosition, locate_target
from .sources import open_source, save_snapshot
from .vision import Vision


def camera_model(config, size):
    if config.camera_calibration:
        model = CameraModel.load(config.camera_calibration)
        if tuple(model.size) != tuple(size):
            raise ValueError(f'Kalibrace kamery je pro {tuple(model.size)}, snímek má {tuple(size)}. '
                             'Kalibrujte ve stejném rozlišení.')
        return model
    width, height = size
    vfov = math.degrees(2*math.atan(height/width*math.tan(math.radians(config.hfov_deg/2))))
    return CameraModel.from_fov(tuple(size), (config.hfov_deg, vfov))


def drone_position(config):
    if config.drone_pose:
        return DronePosition(*config.drone_pose)
    if config.pose_file:
        return DronePosition.from_json(config.pose_file)
    return None


def build_result(result, config):
    """Úhel na tečku + (s polohou dronu) její souřadnice. Chyba souboru s polohou se jen ohlásí."""
    record = dict(angles_deg=dict(right=result.right, forward=result.forward), samples=result.samples,
                  spread_deg=result.spread_deg, track_id=result.track_id, time_unix=time.time(),
                  drone=None, target=None)
    try:
        drone = drone_position(config)
    except (OSError, ValueError, KeyError) as error:
        record['pose_error'] = str(error)
        return record
    if drone is not None:
        record['drone'] = dict(latitude_deg=drone.latitude_deg, longitude_deg=drone.longitude_deg,
                               height_m=drone.height_m, heading_deg=drone.heading_deg,
                               roll_deg=drone.roll_deg, pitch_deg=drone.pitch_deg)
        target = locate_target(drone, result.right, result.forward)
        record['target'] = target.as_dict() if target else None
    return record


def print_result(record):
    angles = record['angles_deg']
    text = (f"ÚHEL NA STŘED TEČKY: right {angles['right']:+.2f}°, forward {angles['forward']:+.2f}° "
            f"(medián {record['samples']} snímků, rozptyl {record['spread_deg']:.2f}°)")
    target = record.get('target')
    if target:
        text += (f"\nSOUŘADNICE TEČKY: {target['latitude_deg']:.8f}, {target['longitude_deg']:.8f} "
                 f"({target['distance_m']:.2f} m od dronu, azimut {target['bearing_deg']:.0f}°, "
                 f"odhad chyby ±{target['error_m']*100:.0f} cm)")
    elif record.get('drone'):
        text += '\nSOUŘADNICE: osa kamery neprotíná zem dost blízko (kamera míří moc šikmo).'
    if record.get('pose_error'):
        text += f"\nPOLOHA DRONU NEČITELNÁ: {record['pose_error']}"
    print(text, flush=True)


def status_line(aim, observation, fps):
    angles = aim.camera_angles
    text = (f'{aim.state:10s} kamera R={angles.right:+6.2f} F={angles.forward:+6.2f}'
            f' | detekce {observation.processing_ms:4.0f} ms, {fps:4.1f} fps')
    if aim.pixel_error is not None:
        text += f' | odchylka {aim.pixel_error[0]:+6.1f},{aim.pixel_error[1]:+6.1f} px'
    if aim.target_angles is not None:
        text += f' | úhel na tečku R={aim.target_angles.right:+6.2f} F={aim.target_angles.forward:+6.2f}'
    return text


def aim_record(aim, observation, sequence):
    target = aim.target_angles
    return dict(sequence=sequence, sample_time=aim.sample_time, state=aim.state, centered=aim.centered,
                camera_angles_deg=dict(right=aim.camera_angles.right, forward=aim.camera_angles.forward),
                target_angles_deg=None if target is None else dict(right=target.right, forward=target.forward),
                pixel_error=None if aim.pixel_error is None else list(aim.pixel_error),
                track_id=aim.track_id, processing_ms=observation.processing_ms,
                rejections=(observation.red or {}).get('diagnostics', {}).get('rejections'))


def run(config) -> int:
    try:
        if (not config.headless and not config.snapshot and sys.platform.startswith('linux')
                and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))):
            raise RuntimeError('Není grafická plocha (SSH?). Přidejte --headless.')
        with ExitStack() as stack:
            gimbal = None
            if config.gimbal == 'servo' and not config.snapshot:
                from .gimbal import Gimbal, ServoCalibration
                calibration = ServoCalibration.load(config.servo_calibration) if config.servo_calibration else None
                gimbal = stack.enter_context(Gimbal(x_dir=config.servo_x_dir, y_dir=config.servo_y_dir,
                                                    calibration=calibration))
            camera, frame, source = open_source(config, stack)
            if config.snapshot:
                save_snapshot(frame, config.snapshot)
                return 0
            height, width = frame.shape[:2]
            model = camera_model(config, (width, height))
            vision = Vision(config.red_diameter_px)
            detector = vision.detector
            detector.redness_min, detector.r_min = config.redness_min, config.r_min
            detector.red_fraction_min = config.red_fraction
            detector.target_diameter_m, detector.hfov_deg = config.target_diameter_m, config.hfov_deg
            if config.camera_calibration:
                detector.camera_model = model
            aimer = Aimer(model, gimbal=gimbal, image_top=config.image_top, max_speed=config.gimbal_speed,
                          scan=config.scan, fixed_angles=GimbalAngles(*config.fixed_angles),
                          dry_run=config.gimbal == 'dry-run')
            try:
                pose_height = drone_position(config).height_m
            except (OSError, ValueError, KeyError, AttributeError):
                pose_height = None
            # Očekávaná velikost tečky (jen měkké skóre): výška dronu a náklon kamery.
            detector.altitude_source = lambda: {'value_m': pose_height, 'source': 'drone_pose' if pose_height else 'unknown'}
            detector.gimbal_source = lambda: (aimer.angles.right, aimer.angles.forward)
            averager = AngleAverager(config.samples, require_centered=config.gimbal == 'servo')
            stream = stack.enter_context(config.output.open('a', encoding='utf-8')) if config.output else None
            diagnostics = None
            if config.diagnostics:
                from .diagnostics import Diagnostics
                diagnostics = Diagnostics(config.diagnostics)
                print(f'Diagnostika: {diagnostics.directory}', file=sys.stderr)
            preview = None
            if not config.headless:
                from .preview import Preview
                preview = stack.enter_context(Preview(config, vision, camera))
            print('Hledám červenou tečku ... (Ctrl+C ukončí)', file=sys.stderr, flush=True)
            sequence = 0
            last_status = last_result = None
            frames_since = 0
            sample_time = time.monotonic()
            while True:
                observation = vision.observe(frame, sample_time=sample_time)
                aim = aimer.update(observation, sample_time=sample_time)
                sequence += 1
                frames_since += 1
                averager.add(aim)
                now = time.monotonic()
                if last_status is None or now-last_status >= 1:
                    fps = frames_since/(now-last_status) if last_status else 0.
                    print(status_line(aim, observation, fps), flush=True)
                    last_status, frames_since = now, 0
                record = aim_record(aim, observation, sequence)
                if stream is not None:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                if diagnostics is not None:
                    diagnostics.update(frame, observation, aim, record)
                result = averager.result()
                if result is not None and (last_result is None or now-last_result >= 2):
                    result_record = build_result(result, config)
                    print_result(result_record)
                    if config.result:
                        config.result.write_text(json.dumps(result_record, indent=2, allow_nan=False),
                                                 encoding='utf-8')
                    last_result = now
                    if config.once:
                        break
                if config.frames is not None and sequence >= config.frames:
                    break
                if preview is not None and not preview.update(frame, observation, aim):
                    break
                if camera is None:
                    if preview is None:
                        break  # fotka bez okna: jedno vyhodnocení stačí
                    time.sleep(.03)  # fotka v okně: stejný snímek, nové měření se nevytváří
                    continue
                try:
                    frame = camera.read()
                except EOFError:
                    break
                sample_time = time.monotonic()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f'Chyba: {error}', file=sys.stderr)
        return 1
    return 0
