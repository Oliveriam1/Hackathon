"""Spustí kinematickou simulaci přeletu, nikdy neposílá povely dronu."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.approach_simulation import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--moving', action='store_true')
    parser.add_argument('--search', action='store_true', help='Prohledávání syntetického pole po řádcích.')
    parser.add_argument('--no-target', action='store_true', help='Scénář bez detekce cíle.')
    parser.add_argument('--field-bounds', nargs=4, type=float, default=(-10., 10., -10., 10.),
                        metavar=('N_MIN', 'N_MAX', 'E_MIN', 'E_MAX'))
    parser.add_argument('--lane-spacing', type=float, default=4.)
    parser.add_argument('--sensor-radius', type=float, default=3., help='Ideální kruhový dosah detekce v metrech.')
    parser.add_argument('--loss', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--seconds', type=float, default=25.)
    parser.add_argument('--takeoff-height', type=float, help='Vzlet ze země do výšky 1–20 m před přeletem.')
    parser.add_argument('--output', type=Path, help='JSON Lines celé trajektorie.')
    args = parser.parse_args()
    if args.takeoff_height is not None and not (math.isfinite(args.takeoff_height) and 1 <= args.takeoff_height <= 20):
        parser.error('--takeoff-height musí být 1 až 20 m.')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        parser.error('--seconds musí být mezi 0 a 600.')
    if args.loss and not (all(math.isfinite(v) for v in args.loss) and 0 <= args.loss[0] < args.loss[1]):
        parser.error('Neplatný interval výpadku.')
    try:
        rows = simulate(seconds=args.seconds, moving=args.moving, takeoff_height=args.takeoff_height,
                        search=args.search, no_target=args.no_target, field_bounds=args.field_bounds,
                        lane_spacing=args.lane_spacing, sensor_radius=args.sensor_radius,
                        loss_start=args.loss[0] if args.loss else None, loss_end=args.loss[1] if args.loss else None)
    except ValueError as error:
        parser.error(str(error))
    if args.output:
        with args.output.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    for row in rows[::20]:
        altitude = f" vyska={row['height_m']:.2f}m" if row['height_m'] is not None else ''
        search_info = (f" bod={row['search_waypoint']}/{row['search_waypoint_count']} vidi={row['target_visible']}"
                       if args.search else '')
        print(f"t={row['time_s']:5.1f}s  {row['state']:20s}{altitude}{search_info} vzdalenost={row['distance_m']:.2f}m")
    print('SIMULACE:', rows[-1]['state'], f"{rows[-1]['distance_m']:.2f} m od tecky")


if __name__ == '__main__':
    main()
