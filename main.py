"""Command-line entry point for the UAV vision application."""

import argparse
import logging
from pathlib import Path

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
    parser.add_argument("--camera", action="store_true", help="Show the camera and color detections.")
    parser.add_argument("--camera-backend", choices=("opencv", "picamera2"), default="opencv")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--snapshot", type=Path, help="Save one clean frame without a GUI window.")
    parser.add_argument(
        "--white-balance-roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        help="Optional neutral white/gray reference rectangle for color correction.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure standard application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def run_camera(args: argparse.Namespace) -> None:
    """Feed the same balanced BGR frame to both the detector and preview."""
    import cv2

    from src.camera import Camera
    from src.color_balance import WhiteBalance
    from src.detector import ColorDetector

    camera = Camera(args.camera_backend, args.camera_index)
    detector = ColorDetector()
    window_created = False
    try:
        camera.start()
        balance = None
        if args.white_balance_roi is not None:
            camera.lock_white_balance()
            balance = WhiteBalance.from_reference(camera.read(), tuple(args.white_balance_roi))
            logger.info("Measured BGR white-balance gains: %s", balance.gains_bgr)
        while True:
            frame = camera.read()
            if balance is not None:
                frame = balance.apply(frame)
            if args.snapshot is not None:
                if not cv2.imwrite(str(args.snapshot), frame):
                    raise RuntimeError(f"Could not save snapshot: {args.snapshot}")
                logger.info("Clean BGR snapshot saved: %s", args.snapshot)
                return
            green = detector.detect_green(frame)
            red = detector.detect_red(frame)
            centering = detector.get_centering(frame, green)
            overlay = frame.copy()
            for label, detections, color in (
                ("GREEN", [green] if green else [], (0, 255, 0)),
                ("RED", red, (0, 0, 255)),
            ):
                for detection in detections:
                    x, y, width, height = detection.bounding_box
                    cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
                    cv2.circle(overlay, (detection.center.x, detection.center.y), 4, color, -1)
                    cv2.putText(overlay, label, (x, max(15, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            status = (f"GREEN error=({centering.error_x},{centering.error_y}) centered={centering.centered}"
                      if centering.detected else "GREEN: NOT DETECTED")
            cv2.putText(overlay, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("UAV Vision", overlay)
            window_created = True
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("UAV Vision", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            camera.close()
        finally:
            if window_created:
                cv2.destroyAllWindows()


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
        if args.camera or args.snapshot is not None:
            if args.dry_run:
                logger.info("Dry-run: camera access skipped")
            else:
                run_camera(args)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0
    except Exception:
        logger.exception("Application failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
