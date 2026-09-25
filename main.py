"""Míření kamery na střed červené tečky (a volitelně výpočet jejích souřadnic).

  python3 main.py --picamera 0 --headless                         # jen úhel na tečku
  python3 main.py --picamera 0 --headless --drone-pose 50.0875123 14.4213456 5.0 0
Všechny parametry: python3 main.py --help
"""
from src.config import parse_args
from src.app import run


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
