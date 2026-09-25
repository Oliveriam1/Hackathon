"""Detekce červené tečky, míření kamery na její střed, souřadnice a telemetrie.

  python3 main.py --jako-red-tracker --headless --status --altitude 5            # jen detekce
  python3 main.py --jako-red-tracker --headless --aim --altitude 5               # + serva míří na střed tečky
  python3 main.py --jako-red-tracker --headless --aim --drone-pose 50.0875 14.4213 5 0 --send 192.168.1.20:5005
Ctrl+C ukončí.
"""

import argparse
import json
import math
import sys
import os
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.pi_camera import PiCamera
from src.vision import Vision, annotate_observation
from src.publisher import JSONPublisher, detection_record
from src.target_lock import TargetLock
from src.sender import TelemetrySender
from src.diagnostics import Diagnostics
from src.aim import Aimer, AngleAverager
from src.geolocation import CameraModel, GimbalAngles
from src.locate import DronePosition, locate_target


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    source.add_argument("--image", type=Path, help="Obrázek místo živé kamery.")
    source.add_argument("--video", type=Path, help="Videozáznam místo živé kamery, např. z letu.")
    source.add_argument("--demo", action="store_true", help="Testovací obraz bez kamery.")
    source.add_argument("--picamera", type=int, metavar="INDEX", help="CSI kamera přes Picamera2, např. --picamera 0.")
    parser.add_argument("--snapshot", type=Path, help="Uloží jeden snímek bez grafického okna (např. test.jpg).")
    parser.add_argument('--headless', action='store_true', help='Jen data, bez grafického okna; ukončení Ctrl+C.')
    parser.add_argument('--status', action='store_true', help='Čitelný stav 1x za sekundu místo JSON v terminálu.')
    parser.add_argument('--diagnostics', type=Path, help='Uloží nejvýše 30 dvojic raw/marked snímků, každé 2 s.')
    parser.add_argument('--sensitivity', type=float, choices=(1, 1.5, 2, 3), default=1.5)
    parser.add_argument('--hough', action='store_true', help='Pomalá záloha pro přerušené kruhové hrany (pro porovnání).')
    parser.add_argument('--detector', choices=('geometry', 'red'), default='red',
                        help='red (výchozí) = červená tečka 1:1 jako red_tracker.py; geometry = kolečko v obdélníku.')
    parser.add_argument('--altitude', type=float,
                        help='Výška kamery nad zemí v m pro očekávanou velikost tečky (výchozí: výška z '
                             '--drone-pose, jinak 20 m jako red_tracker.py). ZADEJTE SKUTEČNOU VÝŠKU, '
                             'jinak filtr velikosti tečku zahodí.')
    parser.add_argument('--red-diameter-px', type=float, help='Očekávaný průměr červené tečky v pixelech; jinak bez filtru velikosti.')
    parser.add_argument('--tuning-file', help='CSI profil (výchozí ov5647_noir.json: kamera bez IR filtru, jinak růžový obraz). '
                                              'Hodnota none ponechá systémový profil.')
    parser.add_argument('--output', type=Path, help='Připojuje JSON Lines do souboru; jinak zapisuje na stdout.')
    parser.add_argument('--frames', type=int, help='Ukončit po daném počtu nových snímků.')
    parser.add_argument('--width', type=int, default=640, help='Šířka CSI snímku.')
    parser.add_argument('--height', type=int, default=480, help='Výška CSI snímku.')
    parser.add_argument('--mavlink', help='Port/endpoint ArduPilotu; jen telemetrie, např. /dev/ttyACM0.')
    parser.add_argument('--baud', type=int, default=115200, help='Rychlost sériového MAVLink spojení.')
    parser.add_argument('--target-system', type=int, help='Očekávané MAVLink system ID autopilota.')
    parser.add_argument('--ev', type=float, default=None,
                        help='Kompenzace expozice CSI kamery, např. --ev -2. Jinak výchozí nastavení profilu.')
    parser.add_argument('--servo', action='store_true',
                        help='Serva závěsu přes pigpio (BCM 18 a 13) jako red_tracker.py; najedou do 0/0.')
    parser.add_argument('--jako-red-tracker', action='store_true',
                        help='Vše jako red_tracker.py: CSI 1296x972, profil ov5647_noir.json, serva --servo.')
    aim = parser.add_argument_group('míření a souřadnice')
    aim.add_argument('--aim', action='store_true', help='Serva natáčí kameru na střed tečky (zapne --servo).')
    aim.add_argument('--samples', type=int, default=20, help='Počet snímků pro medián úhlu na tečku.')
    pose = aim.add_mutually_exclusive_group()
    pose.add_argument('--drone-pose', nargs=4, type=float, metavar=('LAT', 'LON', 'VYSKA_M', 'KURZ_DEG'),
                      help='Poloha dronu -> spočítat i GPS souřadnice tečky.')
    pose.add_argument('--pose-file', type=Path, help='JSON s polohou dronu (čte se při každém výsledku).')
    aim.add_argument('--result', type=Path, help='Uložit poslední výsledek (úhel, souřadnice) jako JSON.')
    aim.add_argument('--once', action='store_true', help='Skončit po prvním výsledku.')
    aim.add_argument('--send', metavar='IP:PORT', help='Posílat telemetrii přes UDP (např. 192.168.1.20:5005).')
    aim.add_argument('--send-rate', type=float, default=10., help='Telemetrie stavu za sekundu (výchozí 10).')
    aim.add_argument('--servo-calibration', type=Path, help='JSON z tools/calibrate_servos.py.')
    aim.add_argument('--servo-x-dir', type=int, choices=(-1, 1), default=1, help='Otočí směr vnějšího serva.')
    aim.add_argument('--servo-y-dir', type=int, choices=(-1, 1), default=1, help='Otočí směr vnitřního serva.')
    aim.add_argument('--gimbal-speed', type=float, default=15., help='Max. rychlost serv °/s (0-60].')
    aim.add_argument('--no-scan', action='store_true', help='Neprohledávat okolí servy, když tečka není vidět.')
    aim.add_argument('--image-top', choices=('forward', 'right', 'backward', 'left'), default='forward',
                     help='Kam na dronu míří horní okraj obrazu při kameře kolmo dolů.')
    aim.add_argument('--hfov-deg', type=float, default=54.0, help='Vodorovné zorné pole kamery bez kalibrace.')
    aim.add_argument('--camera-calibration', type=Path, help='JSON z calibrate_camera.py (stejné rozlišení).')
    args = parser.parse_args()
    if args.aim:
        args.servo = True
    if args.samples < 3:
        parser.error('--samples musí být alespoň 3.')
    if not 0 < args.gimbal_speed <= 60:
        parser.error('--gimbal-speed musí být v rozsahu (0, 60].')
    if args.aim and (args.image or args.video or args.demo):
        parser.error('--aim potřebuje živou kameru.')
    if args.drone_pose is not None:
        try:
            DronePosition(*args.drone_pose)
        except ValueError as error:
            parser.error(str(error))
    if args.send is not None:
        from src.sender import parse_target
        try:
            parse_target(args.send)
        except ValueError as error:
            parser.error(str(error))
    if args.altitude is None:
        args.altitude = args.drone_pose[2] if args.drone_pose else 20.0
    if args.jako_red_tracker:
        if args.image or args.video or args.demo:
            parser.error('--jako-red-tracker vyžaduje CSI kameru.')
        args.picamera = 0 if args.picamera is None else args.picamera
        args.width, args.height = 1296, 972  # plné zorné pole OV5647, binning 2x2
        args.tuning_file = args.tuning_file or 'ov5647_noir.json'
        args.servo = True
    if args.red_diameter_px is not None and (not np.isfinite(args.red_diameter_px) or args.red_diameter_px <= 0 or args.detector != 'red'):
        parser.error('--red-diameter-px musí být kladné číslo a vyžaduje --detector red.')
    if args.tuning_file is not None and args.picamera is None:
        parser.error('--tuning-file vyžaduje --picamera.')
    if args.hough and args.detector == 'red':
        parser.error('--hough je pouze pro --detector geometry.')
    if not np.isfinite(args.altitude) or args.altitude <= 0:
        parser.error('--altitude musí být kladná výška v metrech.')
    if args.frames is not None and args.frames < 1:
        parser.error('--frames musí být kladné.')
    if args.baud <= 0 or (args.target_system is not None and not 1 <= args.target_system <= 255):
        parser.error('Neplatné --baud nebo --target-system.')
    if args.mavlink and (args.demo or args.image or args.video or args.snapshot):
        parser.error('--mavlink připojujte pouze k živé detekci, nikoliv k záznamu nebo fotografii.')
    if args.width < 1 or args.height < 1:
        parser.error('Rozlišení musí být kladné.')
    if args.output and any(path and path.resolve() == args.output.resolve() for path in (args.image, args.video)):
        parser.error('Výstupní data nesmí přepisovat vstupní obraz/video.')
    if args.snapshot and (args.output or args.headless or args.frames or args.status or args.diagnostics):
        parser.error('--snapshot je samostatný režim fotografie; pro data použijte --headless.')
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")
    if args.picamera is not None and args.picamera < 0:
        parser.error("Index CSI kamery musí být nezáporný.")
    if args.ev is not None and args.picamera is None:
        parser.error('--ev lze použít pouze s --picamera.')
    if args.ev is not None and (not np.isfinite(args.ev) or not -8 <= args.ev <= 8):
        parser.error('--ev musí být v rozsahu -8 až 8.')
    return args


def current_altitude(telemetry, fallback):
    """relative_alt z ArduPilotu, pokud je čerstvá a kladná, jinak --altitude."""
    if telemetry is not None:
        item = telemetry.snapshot()['messages'].get('GLOBAL_POSITION_INT')
        if item and item['age_s'] <= 0.5 and item['values']['relative_alt_m'] > 0.5:
            return item['values']['relative_alt_m']
    return fallback


def camera_model(args, size):
    """Model objektivu: kalibrace z calibrate_camera.py, jinak jmenovité zorné pole."""
    if args.camera_calibration:
        model = CameraModel.load(args.camera_calibration)
        if tuple(model.size) != tuple(size):
            raise ValueError(f'Kalibrace kamery je pro {tuple(model.size)}, snímek má {tuple(size)}.')
        return model
    width, height = size
    vfov = math.degrees(2*math.atan(height/width*math.tan(math.radians(args.hfov_deg/2))))
    return CameraModel.from_fov(tuple(size), (args.hfov_deg, vfov))


def aim_record(aim):
    target = aim.target_angles
    return dict(state=aim.state, centered=aim.centered, sample_time=aim.sample_time,
                camera_angles_deg=dict(right=aim.camera_angles.right, forward=aim.camera_angles.forward),
                target_angles_deg=None if target is None else dict(right=target.right, forward=target.forward),
                pixel_error=None if aim.pixel_error is None else list(aim.pixel_error))


def build_result(result, args):
    """Zprůměrovaný úhel na tečku + souřadnice, když je známa poloha dronu."""
    record = dict(angles_deg=dict(right=result.right, forward=result.forward), samples=result.samples,
                  spread_deg=result.spread_deg, time_unix=time.time(), drone=None, target=None)
    try:
        drone = (DronePosition(*args.drone_pose) if args.drone_pose else
                 DronePosition.from_json(args.pose_file) if args.pose_file else None)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
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
        text += '\nSOUŘADNICE: osa kamery míří příliš šikmo, zem neprotne dost blízko.'
    if record.get('pose_error'):
        text += f"\nPOLOHA DRONU NEČITELNÁ: {record['pose_error']}"
    print(text, file=sys.stderr, flush=True)


def draw_aim(image, aim):
    if not aim:
        return
    text = aim['state']
    if aim['target_angles_deg']:
        text += f" uhel R={aim['target_angles_deg']['right']:+.2f} F={aim['target_angles_deg']['forward']:+.2f}"
    cv2.putText(image, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3)
    cv2.putText(image, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)


def main() -> int:
    args = parse_args()
    window_name = "Kamera - Q / Esc: konec"
    window_created = False
    try:
        if not args.snapshot and not args.headless and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('Není dostupná grafická plocha. Přes SSH použijte --snapshot test.jpg.')
        with ExitStack() as stack:
            telemetry = None
            if args.mavlink:
                from src.telemetry import MAVLinkTelemetry
                telemetry = stack.enter_context(MAVLinkTelemetry(args.mavlink, args.baud, args.target_system))
            gimbal = None
            if args.servo:
                # red_tracker.py zapíná serva před kamerou.
                from src.gimbal import Gimbal, ServoCalibration
                calibration = ServoCalibration.load(args.servo_calibration) if args.servo_calibration else None
                try:
                    gimbal = stack.enter_context(Gimbal(x_dir=args.servo_x_dir, y_dir=args.servo_y_dir,
                                                        calibration=calibration))
                except RuntimeError as error:
                    if not args.jako_red_tracker or args.aim:
                        raise
                    print(f'Varování: pokračuji bez serv. {error}', file=sys.stderr)
            camera = None
            if args.demo:
                frame = np.full((480, 640, 3), 35, dtype=np.uint8)
                cv2.rectangle(frame, (100, 100), (220, 220), (0, 200, 0), -1)
                cv2.circle(frame, (460, 300), 45, (0, 0, 230), -1)
                cv2.rectangle(frame, (370, 210), (570, 400), (220, 220, 220), 3)
            elif args.image is not None:
                frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Soubor nelze načíst jako obrázek.")
            elif args.video is not None:
                if not args.video.is_file():
                    raise RuntimeError("Videosoubor neexistuje.")
                camera = stack.enter_context(Camera(str(args.video)))
                frame = camera.read()
            else:
                device = PiCamera(args.picamera, ev=args.ev,
                                  width=args.width, height=args.height,
                                  tuning_file=(None if args.tuning_file == 'none' else
                                               args.tuning_file or 'ov5647_noir.json')) if args.picamera is not None else Camera(args.camera)
                camera = stack.enter_context(device)
                frame = camera.read()
            received_at = time.time()
            sample_time = time.monotonic()
            if args.snapshot:
                extension = args.snapshot.suffix.lower()
                if extension not in ('.jpg', '.jpeg', '.png'):
                    raise RuntimeError('Snímek musí mít příponu .jpg, .jpeg nebo .png.')
                success, encoded = cv2.imencode(extension, frame)
                if not success:
                    raise RuntimeError('Snímek se nepodařilo zakódovat.')
                encoded.tofile(args.snapshot)
                print(f'Snímek uložen: {args.snapshot.resolve()}')
                return 0
            vision = Vision(mode=args.detector, expected_diameter=args.red_diameter_px)
            if args.detector == 'red':
                vision.detector.altitude_source = lambda: current_altitude(telemetry, args.altitude)
                if gimbal is not None:
                    vision.detector.gimbal_source = lambda: (gimbal.x, gimbal.y)
            vision.detector.use_hough = args.hough
            vision.detector.sensitivity = args.sensitivity
            target_lock = TargetLock()
            height_px, width_px = frame.shape[:2]
            model = camera_model(args, (width_px, height_px))
            # Bez --aim serva stojí: úhel na tečku se počítá z polohy tečky v obraze.
            fixed = GimbalAngles(gimbal.x, gimbal.y) if gimbal is not None else GimbalAngles()
            aimer = Aimer(model, gimbal=gimbal if args.aim else None, image_top=args.image_top,
                          max_speed=args.gimbal_speed, scan=not args.no_scan, fixed_angles=fixed)
            averager = AngleAverager(args.samples, require_centered=args.aim)
            sender = stack.enter_context(TelemetrySender(args.send, rate_hz=args.send_rate)) if args.send else None
            last_result = None
            stream = stack.enter_context(args.output.open('a', encoding='utf-8')) if args.output else sys.stdout
            publisher = JSONPublisher(stream) if args.output or not args.status else None
            diagnostics = Diagnostics(stream=sys.stdout if args.status else None, directory=args.diagnostics)
            if diagnostics.directory is not None:
                print(f'Diagnostika: {diagnostics.directory}', file=sys.stderr)
            source_name = 'static' if camera is None else ('video' if args.video else 'camera')
            sequence = 0
            show_edges = False
            if not args.headless:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                window_created = True
                print('Červené oblasti. E: maska, Q / Escape: konec.' if args.detector == 'red' else
                      'Kolečko v obdélníku. 1/2/3: citlivost, B: 1.5, E: hrany, Q / Escape: konec.', file=sys.stderr)
            observation = None
            while True:
                if observation is None or camera is not None:
                    observation = vision.observe(frame)
                    sequence += 1
                    record = detection_record(observation, sequence=sequence,
                                              received_at=received_at, source=source_name)
                    record['telemetry'] = telemetry.snapshot() if telemetry is not None else None
                    record['visual_lock'] = target_lock.update(observation, sample_time=sample_time,
                                                              now=time.monotonic())
                    aim_state = aimer.update(observation, sample_time=sample_time)
                    averager.add(aim_state)
                    record['aim'] = aim_record(aim_state)
                    if sender is not None:
                        sender.send_aim(record['aim'])
                    result = averager.result()
                    now = time.monotonic()
                    if result is not None and (last_result is None or now-last_result >= 2):
                        result_record = build_result(result, args)
                        record['result'] = result_record
                        print_result(result_record)
                        if sender is not None:
                            sender.send_result(result_record)
                        if args.result:
                            args.result.write_text(json.dumps(result_record, indent=2, ensure_ascii=False,
                                                              allow_nan=False), encoding='utf-8')
                        last_result = now
                    if publisher is not None:
                        publisher.publish(record)
                    diagnostics.update(frame, observation, record)
                    if args.once and record.get('result'):
                        break
                if args.frames is not None and sequence >= args.frames:
                    break
                if args.headless:
                    if camera is None:
                        break  # Statický obrázek není několik nezávislých měření.
                    try:
                        frame = camera.read()
                    except EOFError:
                        break
                    received_at = time.time()
                    sample_time = time.monotonic()
                    continue
                background = cv2.cvtColor(vision.detector.edges, cv2.COLOR_GRAY2BGR) if show_edges else frame
                image = annotate_observation(background, observation)
                draw_aim(image, record.get('aim'))
                cv2.imshow(window_name, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('1'), ord('2'), ord('3')):
                    vision.detector.sensitivity = int(chr(key))
                if key in (ord('b'), ord('B')):
                    vision.detector.sensitivity = 1.5
                if camera is None and key in (ord('1'), ord('2'), ord('3'), ord('b'), ord('B')):
                    vision.reset()
                    observation = None
                if key in (ord('e'), ord('E')):
                    show_edges = not show_edges
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if camera is not None:
                    try:
                        frame = camera.read()
                        received_at = time.time()
                        sample_time = time.monotonic()
                    except EOFError:
                        break  # konec záznamu
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Chyba náhledu: {error}", file=sys.stderr)
        return 1
    finally:
        if window_created:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
