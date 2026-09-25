"""Vstupní bod: načtení konfigurace a spuštění aplikace."""
from src.config import parse_args
from src.app import run


def main(argv=None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
