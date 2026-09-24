"""NAVIGATION entry point: JSON centering measurements -> informational state."""

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

from src.models import CenteringResult, Point
from src.navigation import Navigation, centering_from_dict
from src.publisher import ConsolePublisher

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GREEN centering state machine; no flight commands")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="JSON-lines file containing VisionResult or centering; '-' for stdin.")
    source.add_argument("--dry-run", action="store_true", help="Run an explicitly synthetic centering example.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(message)s")
    navigation, publisher = Navigation(), ConsolePublisher()

    def report(centering: CenteringResult | None) -> None:
        previous = navigation.state
        decision = navigation.update(centering)
        if decision.state != previous or centering is None:
            logger.info("STATE: %s", decision.state.value)
        logger.debug("X error: %+.3f; Y error: %+.3f",
                     decision.horizontal_error, decision.vertical_error)
        publisher.publish(decision)

    try:
        if args.dry_run:
            logger.info("Dry-run: synthetic centering input")
            report(CenteringResult(True, Point(435, 525), -65, 25, -0.13, 0.05, False))
        elif args.input is None:
            report(None)
        else:
            context = nullcontext(sys.stdin) if args.input == "-" else Path(args.input).open(encoding="utf-8")
            received = False
            with context as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    received = True
                    try:
                        report(centering_from_dict(json.loads(line)))
                    except (ValueError, TypeError, KeyError, AttributeError):
                        logger.warning("Invalid centering input; resetting to IDLE")
                        report(None)
            if not received:
                report(None)
        return 0
    except OSError as exc:
        logger.warning("Navigation input unavailable: %s", exc)
        report(None)
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except Exception:
        logger.exception("Navigation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
