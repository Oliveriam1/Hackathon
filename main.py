"""Detekce terče: JSON Lines a volitelný diagnostický náhled. Ctrl+C ukončí sběr."""

import argparse
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
from src.diagnostics import Diagnostics


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
    parser.add_argument('--altitude', type=float, default=20.0,
                        help='Výška nad zemí v m pro očekávanou velikost tečky (jako red_tracker.py 20 m); '
                             's --mavlink se použije relative_alt z ArduPilotu.')
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
    args = parser.parse_args()
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
                from src.gimbal import Gimbal
                try:
                    gimbal = stack.enter_context(Gimbal())
                except RuntimeError as error:
                    if not args.jako_red_tracker:
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
                    record['autonomy'] = {'enabled': False, 'state': 'NOT_IMPLEMENTED',
                                          'missing': ['flight_adapter', 'zone_boundary', 'gimbal_feedback',
                                                      'exposure_telemetry_synchronization']}
                    if publisher is not None:
                        publisher.publish(record)
                    diagnostics.update(frame, observation, record)
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
