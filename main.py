"""Circle-in-quadrilateral camera preview or a simulated scan-and-report mission."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import logging

import cv2
import numpy as np

from src.pi_camera import PiCamera
from src.target_detector import CircleInQuadrilateralDetector, annotate


def make_demo() -> np.ndarray:
    """Synthetic grayscale-like scene used only to verify the detector without hardware."""
    image = np.full((720, 1280, 3), 115, dtype=np.uint8)
    quad = np.array([[270, 150], [1020, 185], [930, 610], [220, 565]], np.int32)
    cv2.polylines(image, [quad], True, (220, 220, 220), 8)
    cv2.circle(image, (620, 380), 72, (35, 35, 35), 8)
    return image


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Cannot load image: {path}")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Test one image instead of the Pi camera.")
    source.add_argument("--demo", action="store_true", help="Run on a generated test scene.")
    parser.add_argument("--camera-index", type=int, default=0, help="Picamera2 camera index (default: 0).")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--snapshot", type=Path, help="Save one annotated frame and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the complete mission without UAV/camera hardware.")
    parser.add_argument("--mission", type=Path, help="Mission JSON (flight adapter required outside dry-run).")
    parser.add_argument("--udp-host", help="Explicit destination PC IPv4 address for mission JSON.")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run_mission(args: argparse.Namespace) -> int:
    from src.config import load_mission_config
    from src.mission import MissionController
    from src.models import MissionState
    from src.publisher import ConsolePublisher, UDPPublisher
    from src.simulation import SimulatedFlightController, StaticImageCamera

    if not args.dry_run:
        raise RuntimeError("Real flight-controller adapter is not configured. Use --dry-run to simulate the mission.")
    if args.snapshot:
        raise ValueError("--snapshot belongs to camera preview, not the mission")
    config_path = args.mission or Path(__file__).parent / "config" / "mission.example.json"
    config = load_mission_config(config_path)
    frame = load_image(args.image) if args.image is not None else make_demo()
    logging.info("SIMULATED MISSION: no flight commands will reach hardware")
    publisher = UDPPublisher(args.udp_host, args.udp_port) if args.udp_host else ConsolePublisher()
    controller = MissionController(config, SimulatedFlightController(config.start),
                                   StaticImageCamera(frame), CircleInQuadrilateralDetector(), publisher)
    report = controller.run()
    if report.error:
        logging.error("Mission stopped: %s", report.error)
    return 0 if report.state == MissionState.COMPLETE else 1


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s %(message)s")
    if args.dry_run or args.mission is not None:
        try:
            return run_mission(args)
        except KeyboardInterrupt:
            return 130
        except Exception:
            logging.exception("Mission initialization failed")
            return 1
    if args.udp_host:
        logging.error("--udp-host is only used with a mission")
        return 1
    detector = CircleInQuadrilateralDetector()
    camera = None
    window_name = "IR circle in quadrilateral - Q/Esc to exit"

    try:
        if args.demo:
            frame = make_demo()
        elif args.image is not None:
            frame = load_image(args.image)
        else:
            camera = PiCamera(args.camera_index, width=args.width, height=args.height)
            camera.open()
            frame = camera.read()

        if args.snapshot:
            result = detector.detect(frame)
            output = annotate(frame, result)
            suffix = args.snapshot.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Snapshot must end in .jpg, .jpeg or .png")
            ok, encoded = cv2.imencode(suffix, output)
            if not ok:
                raise RuntimeError("Could not encode snapshot.")
            encoded.tofile(args.snapshot)
            if result.targets:
                for index, target in enumerate(result.targets, 1):
                    print(
                        f"TARGET {index}: x={target.center.x:.1f}, y={target.center.y:.1f}, "
                        f"confidence={target.confidence:.2f}"
                    )
            else:
                print("NO TARGET")
            print(f"Saved: {args.snapshot.resolve()}")
            return 0

        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("No graphical display. Use --snapshot output.jpg when running headless/SSH.")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            result = detector.detect(frame)
            output = annotate(frame, result)
            cv2.putText(
                output,
                f"targets: {len(result.targets)} | quads: {result.quadrilateral_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if camera is not None:
                frame = camera.read()
            else:
                # Static image/demo only needs one displayed frame.
                if key != 255:
                    break

    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
