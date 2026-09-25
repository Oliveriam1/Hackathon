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
    parser.add_argument('--lane-spacing', type=float, default=4.)
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--mission-timeout', type=float, default=600.)
    parser.add_argument('--no-target', action='store_true')
    parser.add_argument('--moving', action='store_true')
    parser.add_argument('--target-loss', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--fault', choices=('telemetry', 'map', 'camera', 'camera_unlocked', 'manual', 'stop', 'reject_arm', 'no_climb'))
    parser.add_argument('--fault-at', type=float, default=20.)
    parser.add_argument('--output', type=Path, help='JSONL celé mise; přepíše zadaný soubor.')
    args = parser.parse_args()
    try:
        settings = MissionSettings(StartReference(*args.start), tuple(args.field_bounds),
                                   args.height, args.lane_spacing, args.mission_timeout)
        rows = simulate_mission(settings, seconds=args.seconds, no_target=args.no_target,
                                moving=args.moving, fault=args.fault, fault_at=args.fault_at,
                                target_loss=args.target_loss)
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
    if rows[-1]['mission']['target_result']:
        print(json.dumps(rows[-1]['mission']['target_result'], allow_nan=False))


if __name__ == '__main__':
    main()
