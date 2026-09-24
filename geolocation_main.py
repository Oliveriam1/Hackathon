"""GEOLOCATION entry point: a pixel plus calibration/telemetry -> ground coordinate."""

import argparse
import logging
from pathlib import Path

from src.config import CONFIG_DIR, load_camera_calibration, read_json
from src.models import Attitude, CameraIntrinsics, CameraMount, GeoCoordinate, Point, UAVTelemetry
from src.publisher import ConsolePublisher
from src.telemetry import telemetry_from_dict

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pixel-to-WGS84 math; no detection or navigation")
    parser.add_argument("--pixel", nargs=2, type=int, metavar=("U", "V"))
    parser.add_argument("--calibration", type=Path, help="JSON camera intrinsics and mount.")
    parser.add_argument("--telemetry", type=Path, help="Explicit telemetry snapshot matching the frame.")
    parser.add_argument("--dry-run", action="store_true", help="Run a labelled synthetic nadir example.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        from src.geolocation import TargetGeolocator

        if args.dry_run:
            if args.pixel is not None or args.calibration is not None or args.telemetry is not None:
                raise ValueError("--dry-run uses only synthetic inputs; do not combine with real inputs")
            logger.info("Dry-run: SYNTHETIC calibration, nadir mount, pixel and telemetry")
            intrinsics = CameraIntrinsics(500, 500, 500, 500, 1000, 1000)
            mount = CameraMount(0, -90, 0)
            pixel = Point(500, 500)
            telemetry = UAVTelemetry(GeoCoordinate(50, 14), 10, Attitude(0, 0, 0))
        else:
            intrinsics, mount = load_camera_calibration(
                args.calibration if args.calibration is not None else CONFIG_DIR / "camera_calibration.json"
            )
            if args.telemetry is None:
                raise ValueError("TELEMETRY REQUIRED: provide --telemetry with explicit altitude_agl_m")
            telemetry = telemetry_from_dict(read_json(args.telemetry))
            if args.pixel is None:
                raise ValueError("PIXEL REQUIRED: provide --pixel U V")
            pixel = Point(*args.pixel)
        result = TargetGeolocator(intrinsics, mount).geolocate(pixel, telemetry)
        logger.info("Localization: North=%+.3f m East=%+.3f m lat=%.8f lon=%.8f",
                    result.north_offset_m, result.east_offset_m,
                    result.coordinate.latitude, result.coordinate.longitude)
        ConsolePublisher().publish({"synthetic": args.dry_run, "pixel": pixel, "location": result})
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        logger.error("Geolocation unavailable: %s", exc)
        return 1
    except Exception:
        logger.exception("Geolocation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
