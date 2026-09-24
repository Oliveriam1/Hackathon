"""Command-line entry point for the UAV vision application."""

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="UAV vision application")
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG logging."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without connecting to UAV or Raspberry Pi hardware.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure standard application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def main() -> int:
    """Initialize the application and return its exit status."""
    args = parse_args()
    configure_logging(args.debug)

    try:
        logger.info("UAV Vision starting")

        import src

        logger.info("Version: %s", src.__version__)
        logger.info("Mode: %s", "dry-run" if args.dry_run else "normal")
        logger.info("Initialization complete")
    except Exception:
        logger.exception("Application initialization failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
