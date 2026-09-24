"""UAV vision entry point; hardware access requires an explicit --camera flag."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from src.detector import ColorDetector
    from src.geolocation import TargetGeolocator
    from src.models import UAVTelemetry
    from src.telemetry import TelemetryProvider

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
    parser.add_argument("--camera", action="store_true", help="Open a local debug camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--calibration", type=Path, help="JSON file with camera intrinsics and mount."
    )
    parser.add_argument(
        "--mock-telemetry", action="store_true",
        help="Use clearly labelled STATIC TEST telemetry, never live UAV data.",
    )
    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    """Configure standard application logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def draw_debug_overlay(
    frame: np.ndarray,
    detector: ColorDetector,
    geolocator: TargetGeolocator | None = None,
    telemetry: UAVTelemetry | None = None,
    *,
    mock_telemetry: bool = False,
) -> np.ndarray:
    """Detect colors and annotate a frame copy; missing geolocation inputs are OK."""
    import cv2

    green = detector.detect_green(frame)
    red = detector.detect_red(frame)
    centering = detector.get_centering(frame, green)
    overlay = frame.copy()
    lines: list[str] = []
    if mock_telemetry:
        lines.append("STATIC TEST TELEMETRY - NOT REAL TARGET COORDINATES")

    unavailable: str | None = None
    if geolocator is None:
        lines.append("CAMERA CALIBRATION REQUIRED")
        unavailable = "GEOLOCATION: CAMERA NOT CALIBRATED"
    elif frame.shape[:2] != (geolocator.intrinsics.height, geolocator.intrinsics.width):
        unavailable = "GEOLOCATION: CALIBRATION SIZE MISMATCH"
    if telemetry is None:
        lines.append("GEOLOCATION: NO TELEMETRY")
        if unavailable is None:
            unavailable = "GEOLOCATION: NO TELEMETRY"
    if unavailable is not None and unavailable not in lines:
        lines.append(unavailable)

    if green is None:
        lines.append("GREEN: NOT DETECTED")
    else:
        x, y, width, height = green.bounding_box
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.circle(overlay, (green.center.x, green.center.y), 4, (0, 255, 0), -1)
        lines.append(
            f"GREEN error=({centering.error_x:+d},{centering.error_y:+d}) "
            f"centered={centering.centered}"
        )

    for number, detection in enumerate(red, start=1):
        x, y, width, height = detection.bounding_box
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.circle(overlay, (detection.center.x, detection.center.y), 4, (0, 0, 255), -1)
        cv2.putText(overlay, f"RED {number}", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        lines.append(f"RED {number} pixel=({detection.center.x},{detection.center.y})")
        if unavailable is not None:
            continue
        assert geolocator is not None and telemetry is not None
        try:
            location = geolocator.geolocate(detection.center, telemetry)
        except ValueError as exc:
            lines.append(f"GEOLOCATION: UNAVAILABLE ({exc})")
            logger.debug("RED %s cannot be geolocated: %s", number, exc)
            continue
        lines.append(
            f"North: {location.north_offset_m:+.2f} m  "
            f"East: {location.east_offset_m:+.2f} m"
        )
        lines.append(
            f"lat={location.coordinate.latitude:.8f} "
            f"lon={location.coordinate.longitude:.8f}"
        )
        logger.debug("RED %s: %s", number, location)

    for index, line in enumerate(lines):
        position = (8, 20 + index * 20)
        cv2.putText(overlay, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def run_camera(
    camera_index: int,
    geolocator: TargetGeolocator | None,
    telemetry_provider: TelemetryProvider | None = None,
    *,
    mock_telemetry: bool = False,
) -> None:
    """Local debug loop; q/Escape or closing the window stops capture."""
    import cv2

    from src.camera import OpenCVCamera
    from src.detector import ColorDetector

    detector = ColorDetector()
    intrinsics = geolocator.intrinsics if geolocator is not None else None
    logger.info("Camera debug mode; press q or Escape to stop")
    try:
        with OpenCVCamera(camera_index, intrinsics) as camera:
            while True:
                frame = camera.read()
                telemetry = None
                if telemetry_provider is not None:
                    try:
                        telemetry = telemetry_provider.get_telemetry()
                    except Exception:
                        logger.debug("Telemetry unavailable", exc_info=True)
                overlay = draw_debug_overlay(
                    frame, detector, geolocator, telemetry, mock_telemetry=mock_telemetry
                )
                cv2.imshow("UAV Vision", overlay)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if cv2.getWindowProperty("UAV Vision", cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
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

        geolocator = None
        if args.calibration is not None:
            from src.camera import load_camera_calibration
            from src.geolocation import TargetGeolocator

            intrinsics, mount = load_camera_calibration(args.calibration)
            geolocator = TargetGeolocator(intrinsics, mount)
            logger.info("Camera calibration loaded: %s", args.calibration)
        else:
            logger.info("CAMERA CALIBRATION REQUIRED")

        telemetry_provider = None
        if args.mock_telemetry:
            from src.models import Attitude, GeoCoordinate, UAVTelemetry
            from src.telemetry import StaticTelemetryProvider

            telemetry_provider = StaticTelemetryProvider(
                UAVTelemetry(GeoCoordinate(50.0, 14.0), 10.0, Attitude(0.0, 0.0, 0.0))
            )
            logger.warning("STATIC TEST TELEMETRY enabled; coordinates are demonstration only")
        else:
            logger.info("GEOLOCATION: NO TELEMETRY")

        logger.info("Initialization complete")
        if args.camera and args.dry_run:
            logger.info("Dry-run: camera access skipped")
        elif args.camera:
            run_camera(
                args.camera_index, geolocator, telemetry_provider,
                mock_telemetry=args.mock_telemetry,
            )
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        return 0
    except Exception:
        logger.exception("Application failed")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
