"""Celá mise na simulačním backendu, bez připojení dronu."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import StartReference
from src.mission_controller import MissionSettings
from src.mission_simulation import simulate_mission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('simulation',), default='simulation')
    parser.add_argument('--start', nargs=2, type=float, default=(50., 14.), metavar=('LAT', 'LON'),
                        help='Výchozí souřadnice jsou fiktivní, pouze pro simulaci.')
    parser.add_argument('--height', type=float, default=5.)
    parser.add_argument('--field-bounds', nargs=4, type=float, default=(-10., 10., -10., 10.))
    parser.add_argument('--lane-spacing', type=float, default=3.)
    parser.add_argument('--max-speed', type=float, default=1., help='Nejvyšší vodorovná rychlost mise, m/s.')
    parser.add_argument('--target', nargs=2, type=float, default=(8., 5.), metavar=('NORTH', 'EAST'))
    parser.add_argument('--tape', nargs=4, type=float, action='append', default=[], metavar=('N1', 'E1', 'N2', 'E2'),
                        help='Úsek skutečné pásky v simulaci (m od startu); lze opakovat.')
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--mission-timeout', type=float, default=600.)
    parser.add_argument('--no-target', action='store_true')
    parser.add_argument('--moving', action='store_true')
    parser.add_argument('--target-loss', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--fault', choices=('telemetry', 'map', 'camera', 'camera_unlocked', 'manual', 'stop', 'reject_arm', 'no_climb'))
    parser.add_argument('--fault-at', type=float, default=20.)
    parser.add_argument('--precision-loops', type=int, default=2, help='Měření červená+zelená po nalezení (0 = vypnuto).')
    parser.add_argument('--gps-drift', type=float, default=.5, help='Směrodatná odchylka ujíždění GPS/EKF v m.')
    parser.add_argument('--camera-noise', type=float, default=.05, help='Šum polohy bodu z kamery v m.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--finish', choices=('hold', 'land'), default='hold')
    parser.add_argument('--output', type=Path, help='JSONL celé mise; přepíše zadaný soubor.')
    args = parser.parse_args()
    try:
        settings = MissionSettings(StartReference(*args.start), tuple(args.field_bounds),
                                   args.height, args.lane_spacing, args.mission_timeout, args.max_speed,
                                   precision_loops=args.precision_loops, finish=args.finish)
        rows = simulate_mission(settings, seconds=args.seconds, no_target=args.no_target,
                                moving=args.moving, fault=args.fault, fault_at=args.fault_at,
                                target_loss=args.target_loss,
                                tapes=[((t[0], t[1]), (t[2], t[3])) for t in args.tape], target_at=tuple(args.target),
                                gps_drift_m=args.gps_drift, camera_noise_m=args.camera_noise, seed=args.seed)
        if args.output:
            with args.output.open('w', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    previous = None
    last_report = -10.
    for row in rows:
        state = row['command']['state']
        if state != previous and row['time_s']-last_report >= .5:
            print(f"t={row['time_s']:6.1f}s {state:22s} vyska={row['height_m']:.2f}m "
                  f"cil={row['distance_m']:.2f}m duvod={row['command']['reason']}")
            previous, last_report = state, row['time_s']
    print('SIMULACE:', rows[-1]['command']['state'])
    if rows[-1]['tape_lines']:
        print('PASKA (sever1, vychod1, sever2, vychod2):', [[round(v, 2) for v in l] for l in rows[-1]['tape_lines']])
        print(f"nejsevernejsi poloha: {max(r['north_m'] for r in rows):.2f} m")
    result = rows[-1]['mission']['target_result']
    if result:
        if result.get('method') == 'green_relative':
            n, e = result['north_from_green_m'], result['east_from_green_m']
        else:
            n, e = result['north_m'], result['east_m']
        tn, te = args.target
        print(f"VÝSLEDEK ({result['method']}): S {n:+.3f} m, V {e:+.3f} m od zelené; "
              f"skutečnost S {tn:+.3f}, V {te:+.3f}; CHYBA {((n-tn)**2+(e-te)**2)**.5*100:.1f} cm")


if __name__ == '__main__':
    main()
