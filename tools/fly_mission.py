"""Skutečná mise: vzlet, hledání po poli, páska = zakázaná zóna, přílet nad tečku.

Příklady:
  # ArduPilot SITL bez kamery (ověří vzlet, trasu, hranice, ACK a převzetí):
  python3 tools/fly_mission.py --connect tcp:127.0.0.1:5760 --field-bounds -10 10 -10 10 --fake-camera

  # Skutečný dron (Pi <-> letový kontrolér přes UART, serva kamery):
  python3 tools/fly_mission.py --connect /dev/serial0 --baud 921600 --real-flight \
      --field-bounds -10 10 -10 10 --takeoff-height 5 --output mise.jsonl

Start (0, 0) = místo, kde dron stojí při spuštění. --field-bounds jsou metry
sever/východ od něj a musí odpovídat skutečnému poli (a geofence v ArduPilotu).
Ukončení: Ctrl+C (zabrzdí a přepne do --exit-mode) nebo přepnutí režimu na
vysílačce (program okamžitě přestane posílat povely).
"""
import argparse
import shlex
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mission_controller import MissionController, MissionSettings, tape_safe_speed
from src.mavlink_backend import ArduPilotBackend, connect


def camera_argv(args):
    argv = ['--picamera', str(args.picamera), '--width', str(args.cam_width), '--height', str(args.cam_height),
            '--headless', '--detect-boundary', '--locate-target', '--ground-relative-alt', '0',
            '--mavlink', args.connect]
    if args.gimbal == 'servo':
        argv.append('--track-camera')
    else:
        argv += ['--camera-right-deg', '0', '--camera-forward-deg', '0']
    return argv+shlex.split(args.camera_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connect', required=True, help='MAVLink endpoint: /dev/serial0, /dev/ttyACM0, tcp:127.0.0.1:5760, udp:...')
    parser.add_argument('--baud', type=int, default=115200, help='Musí odpovídat SERIALx_BAUD v ArduPilotu.')
    parser.add_argument('--target-system', type=int)
    parser.add_argument('--real-flight', action='store_true',
                        help='Povinné pro sériový port (skutečný dron). tcp:/udp: se bere jako SITL.')
    parser.add_argument('--takeoff-height', type=float, default=5., help='Výška letu nad startem, 1-20 m.')
    parser.add_argument('--field-bounds', nargs=4, type=float, required=True, metavar=('N_MIN', 'N_MAX', 'E_MIN', 'E_MAX'),
                        help='Obdélník pole v metrech sever/východ od startu.')
    parser.add_argument('--lane-spacing', type=float, default=3.)
    parser.add_argument('--max-speed', type=float, default=1., help='Horní mez; dál ji sníží dohled na pásku.')
    parser.add_argument('--timeout', type=float, default=600.)
    parser.add_argument('--green', nargs=2, type=float, metavar=('LAT', 'LON'),
                        help='Známá souřadnice zelené tečky (start). Bez ní se použije GPS startu.')
    parser.add_argument('--precision-loops', type=int, default=2,
                        help='Počet měření červená+zelená po nalezení tečky (0 = jen GPS).')
    parser.add_argument('--green-diameter-m', type=float, default=.2)
    parser.add_argument('--finish', choices=('hold', 'land'), default='hold',
                        help='Po měření: viset nad zelenou, nebo přistát na ní.')
    parser.add_argument('--picamera', type=int, default=0)
    parser.add_argument('--cam-width', type=int, default=1296)
    parser.add_argument('--cam-height', type=int, default=972)
    parser.add_argument('--gimbal', choices=('servo', 'fixed'), default='servo',
                        help='servo = závěs (kolmo dolů při hledání, centrování tečky); fixed = kamera pevně dolů.')
    parser.add_argument('--camera-args', default='', help='Další parametry main.py, např. "--camera-calibration kal.json --ev -1".')
    parser.add_argument('--fake-camera', action='store_true', help='Bez kamery (jen SITL): žádná tečka ani páska.')
    parser.add_argument('--exit-mode', choices=('LOITER', 'RTL', 'LAND', 'NONE'), default='LOITER')
    parser.add_argument('--output', type=Path, help='JSONL log každého kroku mise (20 Hz).')
    parser.add_argument('--yes', action='store_true', help='Bez potvrzení START (jen SITL).')
    args = parser.parse_args()
    simulator = args.connect.startswith(('tcp:', 'udp:', 'udpin:', 'udpout:'))
    if not simulator and not args.real_flight:
        parser.error('Sériový port = skutečný dron. Přidejte --real-flight.')
    if args.fake_camera and not simulator:
        parser.error('--fake-camera je jen pro SITL.')
    if args.yes and not simulator:
        parser.error('Na skutečném dronu je potvrzení START povinné.')

    with ExitStack() as stack:
        pipeline = gimbal = camera = config = None
        if not args.fake_camera:
            from src.config import parse_args
            from src.pipeline import DetectionPipeline
            from src.sources import open_source
            config = parse_args(camera_argv(args))
            if config.servo:
                from src.gimbal import Gimbal
                gimbal = stack.enter_context(Gimbal(x_dir=config.servo_x_dir, y_dir=config.servo_y_dir))
        print(f'MAVLink: připojuji {args.connect} ...', flush=True)
        connection = connect(args.connect, args.baud, target_system=args.target_system)
        stack.callback(connection.close)
        backend = ArduPilotBackend(connection)
        backend.start_io()
        stack.callback(backend.close)
        print("Čekám na telemetrii a EKF origin (max 60 s) ...", flush=True)
        if not backend.wait_ready(60):
            raise SystemExit('Telemetrie nepřichází (HEARTBEAT, LOCAL_POSITION_NED, GLOBAL_POSITION_INT). '
                             'Má autopilot GPS fix a EKF origin?')
        start = backend.capture_origin()
        from src.config import StartReference
        settings = MissionSettings(start, tuple(args.field_bounds), args.takeoff_height,
                                   args.lane_spacing, args.timeout, args.max_speed,
                                   precision_loops=args.precision_loops, finish=args.finish,
                                   green=StartReference(*args.green) if args.green else None)
        controller = MissionController(settings)
        if not args.fake_camera:
            camera, frame, source = open_source(config, stack)
            pipeline = DetectionPipeline(config, telemetry=backend, gimbal=gimbal)
            pipeline.camera_mode = 'SEARCH'
        log = stack.enter_context(args.output.open('w', encoding='utf-8')) if args.output else None
        from src.flight_runner import FlightRunner
        from src.marker_detector import GreenMarkerDetector
        runner = FlightRunner(backend, controller, settings.bounds, pipeline=pipeline, config=config,
                              gimbal=gimbal, log_stream=log, fake_camera=args.fake_camera,
                              green_detector=None if args.fake_camera else GreenMarkerDetector(),
                              green_diameter_m=args.green_diameter_m)

        print(f'Start (0,0): {start.latitude_deg:.7f}, {start.longitude_deg:.7f}')
        print(f'Pole: sever {args.field_bounds[0]}..{args.field_bounds[1]} m, '
              f'východ {args.field_bounds[2]}..{args.field_bounds[3]} m; výška {args.takeoff_height} m; '
              f'rychlost max {controller.speed_cap:.2f} m/s (dohled na pásku {tape_safe_speed(args.takeoff_height):.2f}); '
              f'{len(controller.search.sweep.points)} bodů trasy.')
        if not args.yes:
            print('Pilot drží vysílačku; přepnutí režimu = okamžité převzetí.')
            if input('Napište START pro vzlet: ').strip() != 'START':
                print('Zrušeno.')
                return
        stop = threading.Event()
        thread = threading.Thread(target=runner.mission_loop, args=(stop,), daemon=True, name='mission')
        thread.start()
        last_print = 0.
        try:
            while not stop.is_set():
                if camera is not None:
                    frame = camera.read()
                    runner.camera_step(frame, received_at=time.time(), sample_time=time.monotonic(), source=source)
                else:
                    time.sleep(.05)
                command = runner.last_command
                if command is not None and time.monotonic()-last_print >= 1:
                    vehicle, height, _ = backend.vehicle()
                    snap = controller.snapshot()
                    where = (f'N={vehicle.position.north_m:6.2f} E={vehicle.position.east_m:6.2f}'
                             if vehicle else 'poloha ?')
                    print(f'{command.state:18s} {command.reason:28s} {where} h={height:5.2f} '
                          f'páska={snap["tape_lines"]} bod={snap["waypoint_index"]}/{snap["waypoint_count"]}',
                          flush=True)
                    last_print = time.monotonic()
                if controller.state in ('MANUAL',):
                    print('Pilot převzal řízení; program končí bez dalších povelů.')
                    break
        except (KeyboardInterrupt, EOFError):
            print('\nZastavení: brzdím.')
            runner.stop_requested = True
            time.sleep(1.)
        finally:
            stop.set()
            thread.join(timeout=2)
            if (controller.state != 'MANUAL' and not backend.land_sent and args.exit_mode != 'NONE'
                    and backend.heartbeat and backend.heartbeat[1]):
                print(f'Přepínám do {args.exit_mode}.')
                backend.set_mode(args.exit_mode)
                time.sleep(.3)
        result = controller.snapshot()['target_result']
        if result and result.get('method') == 'green_relative':
            print(f"ČERVENÁ TEČKA: {result['latitude_deg']:.8f}, {result['longitude_deg']:.8f} (WGS84)")
            print(f"  od zelené: S {result['north_from_green_m']:+.3f} m, V {result['east_from_green_m']:+.3f} m, "
                  f"vzdálenost {result['distance_from_green_m']:.3f} m, azimut {result['bearing_from_green_deg']:.1f}°")
            print(f"  párů měření: {result['pairs']}, rozptyl párů {result['pair_spread_m']:.3f} m, "
                  f"zelená: {'zadaná' if args.green else 'GPS startu (zadejte --green pro přesný výsledek)'}")
        elif result:
            print('TEČKA (jen GPS, bez zelené reference):', result)


if __name__ == '__main__':
    main()
