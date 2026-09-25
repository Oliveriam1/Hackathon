"""Kalibrace serv: kolik stupňů se kamera SKUTEČNĚ natočí na příkaz.

MG996R nemá přesně 90° na 1000 µs a jeho nula nebývá přesně kolmo dolů.
Každý stupeň chyby je při výšce 5 m zhruba 9 cm chyby na zemi, proto se
vyplatí to změřit.

Postup:
  1. Dron postavte vodorovně (vodováha / mobil na rám).
  2. Na kameru přiložte mobil s aplikací sklonoměru (nebo úhloměr).
  3. Spusťte:  python3 tools/calibrate_servos.py --output servo_calibration.json
  4. Program natočí serva postupně do několika poloh a u každé se zeptá,
     jaký úhel naměříte. Zadejte skutečný náklon kamery od svislice
     (doprava / dopředu kladně, doleva / dozadu záporně).
  5. Výsledek použijte:  python3 main.py ... --servo-calibration servo_calibration.json

Když se kamera při kladném příkazu natočí na opačnou stranu, spusťte znovu
s --servo-x-dir -1 nebo --servo-y-dir -1 a stejný parametr pak dejte i main.py.
"""
import argparse
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gimbal import Gimbal, ServoCalibration  # noqa: E402

COMMANDS = (-40., -20., 0., 20., 40.)


def fit(commands, measured):
    """Přímka measured = scale*command + offset metodou nejmenších čtverců."""
    mx, my = statistics.fmean(commands), statistics.fmean(measured)
    sxx = sum((x-mx)**2 for x in commands)
    if sxx == 0:
        raise ValueError('Potřebuji alespoň dvě různé polohy.')
    scale = sum((x-mx)*(y-my) for x, y in zip(commands, measured))/sxx
    offset = my-scale*mx
    residual = max(abs(scale*x+offset-y) for x, y in zip(commands, measured))
    return scale, offset, residual


def ask(prompt):
    while True:
        text = input(prompt).strip().replace(',', '.')
        if text.lower() in ('s', 'skip', ''):
            return None
        try:
            return float(text)
        except ValueError:
            print('  Zadejte číslo ve stupních (nebo Enter pro přeskočení).')


def calibrate_axis(gimbal, axis, commands):
    name = 'DOPRAVA (+) / doleva (-)' if axis == 'right' else 'DOPŘEDU (+) / dozadu (-)'
    print(f'\n=== Osa {axis}: náklon kamery {name} ===')
    used, measured = [], []
    for command in commands:
        if axis == 'right':
            gimbal.move_to(command, 0.)
        else:
            gimbal.move_to(0., command)
        value = ask(f'  příkaz {command:+5.0f}° -> naměřený úhel kamery [°]: ')
        if value is not None:
            used.append(command)
            measured.append(value)
    gimbal.move_to(0., 0.)
    if len(used) < 2:
        raise SystemExit(f'Osa {axis}: málo měření.')
    scale, offset, residual = fit(used, measured)
    if scale < 0:
        print('  POZOR: osa je obrácená – spusťte znovu s opačným --servo-x-dir/--servo-y-dir.')
    print(f'  skutečný = {scale:.3f} * příkaz {offset:+.2f}°   (největší odchylka bodu {residual:.2f}°)')
    if residual > 2:
        print('  POZOR: odchylka nad 2° – servo má vůli nebo se měření netrefilo, zopakujte.')
    return scale, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('servo_calibration.json'))
    parser.add_argument('--servo-x-dir', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--servo-y-dir', type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    with Gimbal(x_dir=args.servo_x_dir, y_dir=args.servo_y_dir) as gimbal:
        print('Kamera najela na 0/0 (má mířit kolmo dolů). Měřte odklon kamery od svislice.')
        right = calibrate_axis(gimbal, 'right', COMMANDS)
        forward = calibrate_axis(gimbal, 'forward', COMMANDS)
    calibration = ServoCalibration(right[0], right[1], forward[0], forward[1])
    calibration.save(args.output)
    print(f'\nUloženo: {args.output}  ->  python3 main.py ... --servo-calibration {args.output}')


if __name__ == '__main__':
    main()
