"""VISION entry point: camera -> color candidates -> JSON; no GNSS or navigation."""

import argparse
import logging
from pathlib import Path

from src.config import CONFIG_DIR, read_json

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera and GREEN/RED candidate detection")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "vision.json")
    parser.add_argument("--camera-backend", choices=("opencv", "picamera2"))
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--camera", "--preview", dest="preview", action="store_true",
                        help="Show the OpenCV debug window; default is headless JSON output.")
    parser.add_argument("--test-pink", action="store_true", help="Temporary pink debug overlay.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without opening hardware.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        from src.camera import create_camera
        from src.detector import ColorDetector
        from src.publisher import ConsolePublisher
        from src.vision import VisionProcessor, run_vision

        config = read_json(args.config)
        backend = args.camera_backend or config["camera_backend"]
        index = args.camera_index if args.camera_index is not None else config["camera_index"]
        camera = create_camera(backend, index, config["width"], config["height"])
        processor = VisionProcessor(ColorDetector(
            min_contour_area=config["min_contour_area"],
            centering_tolerance_px=config["centering_tolerance_px"],
        ))
        if args.max_frames is not None and args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        logger.info("Camera backend: %s", backend)
        if args.dry_run:
            logger.info("Dry-run: configuration valid; camera access skipped")
            return 0
        run_vision(camera, processor, ConsolePublisher(),
                   preview=args.preview or args.test_pink, test_pink=args.test_pink,
                   max_frames=args.max_frames)
        return 0
    except (KeyboardInterrupt, BrokenPipeError):
        return 0
    except Exception:
        logger.exception("Vision failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
