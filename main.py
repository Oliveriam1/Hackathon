"""Legacy Vision launcher; new programs have their own independent entry points."""

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
    parser.add_argument("--camera", action="store_true", help="Open the local camera preview.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--test-pink", action="store_true",
        help="Temporarily include pink detection in the --camera preview.",
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
        logger.info("New entry points: vision_main.py, navigation_main.py, geolocation_main.py")

        import src

        logger.info("Version: %s", src.__version__)
        logger.info("Mode: %s", "dry-run" if args.dry_run else "normal")
        logger.info("Initialization complete")
        if args.camera and args.dry_run:
            logger.info("Dry-run: camera access skipped")
        elif args.camera:
            from src.camera import run_camera_preview

            run_camera_preview(args.camera_index, test_pink=args.test_pink)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0
    except Exception:
        logger.exception("Application failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
