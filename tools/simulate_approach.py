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
    parser.add_argument('--loss', nargs=2, type=float, metavar=('START', 'END'))
    parser.add_argument('--seconds', type=float, default=25.)
    parser.add_argument('--output', type=Path, help='JSON Lines celé trajektorie.')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600:
        parser.error('--seconds musí být mezi 0 a 600.')
    if args.loss and not (all(math.isfinite(v) for v in args.loss) and 0 <= args.loss[0] < args.loss[1]):
        parser.error('Neplatný interval výpadku.')
    rows = simulate(seconds=args.seconds, moving=args.moving,
                    loss_start=args.loss[0] if args.loss else None, loss_end=args.loss[1] if args.loss else None)
    if args.output:
        with args.output.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    for row in rows[::20]:
        print(f"t={row['time_s']:5.1f}s  {row['state']:20s} vzdalenost={row['distance_m']:.2f}m")
    print('SIMULACE:', rows[-1]['state'], f"{rows[-1]['distance_m']:.2f} m od tecky")


if __name__ == '__main__':
    main()
