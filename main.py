"""Živý náhled kamery. Ukončení klávesou Q, Escape nebo zavřením okna."""

import argparse
import sys
import os
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.pi_camera import PiCamera
from src.manual_target import ManualTarget
from src.vision import Vision, annotate_observation
from src.color_calibration import load_profiles
from src.calibration_ui import CalibrationUI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    source.add_argument("--image", type=Path, help="Obrázek místo živé kamery.")
    source.add_argument("--demo", action="store_true", help="Testovací obraz bez kamery.")
    source.add_argument("--picamera", type=int, metavar="INDEX", help="CSI kamera přes Picamera2, např. --picamera 0.")
    parser.add_argument("--snapshot", type=Path, help="Uloží jeden snímek bez grafického okna (např. test.jpg).")
    parser.add_argument('--calibration', type=Path, default=Path(__file__).with_name('color_calibration.json'),
                        help='Soubor barevné kalibrace, načítá se automaticky, S jej uloží.')
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")
    if args.picamera is not None and args.picamera < 0:
        parser.error("Index CSI kamery musí být nezáporný.")

    window_name = "Kamera - Q / Esc: konec"
    window_created = False
    try:
        if not args.snapshot and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('Není dostupná grafická plocha. Přes SSH použijte --snapshot test.jpg.')
        with ExitStack() as stack:
            camera = None
            if args.demo:
                frame = np.full((480, 640, 3), 35, dtype=np.uint8)
                cv2.rectangle(frame, (100, 100), (220, 220), (0, 200, 0), -1)
                cv2.circle(frame, (460, 300), 45, (0, 0, 230), -1)
            elif args.image is not None:
                frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Soubor nelze načíst jako obrázek.")
            else:
                device = PiCamera(args.picamera) if args.picamera is not None else Camera(args.camera)
                camera = stack.enter_context(device)
                frame = camera.read()
            if args.snapshot:
                extension = args.snapshot.suffix.lower()
                if extension not in ('.jpg', '.jpeg', '.png'):
                    raise RuntimeError('Snímek musí mít příponu .jpg, .jpeg nebo .png.')
                success, encoded = cv2.imencode(extension, frame)
                if not success:
                    raise RuntimeError('Snímek se nepodařilo zakódovat.')
                encoded.tofile(args.snapshot)
                print(f'Snímek uložen: {args.snapshot.resolve()}')
                return 0
            target = ManualTarget()
            vision = Vision()
            if args.calibration.exists():
                vision.detector.profiles = load_profiles(args.calibration)
                print(f'Kalibrace načtena: {args.calibration}')
            calibration = CalibrationUI(vision, target, args.calibration)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.setMouseCallback(window_name, calibration.on_mouse)
            print('Kalibrace: G = zelená, C = červená, klik = vzorek, S = uložit, D = výchozí, M = ruční bod.')
            print("Levý klik: označit bod. Pravý klik: zrušit. R: reset reference. Q / Escape: konec.")
            while True:
                calibration.frame = frame
                observation = vision.observe(frame)
                image = annotate_observation(target.annotate(frame), observation)
                cv2.imshow(window_name, calibration.annotate(image))
                key = cv2.waitKey(20) & 0xFF
                calibration.handle_key(key)
                if key in (ord('r'), ord('R')):
                    vision.reset()
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if camera is not None:
                    frame = camera.read()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Chyba náhledu: {error}", file=sys.stderr)
        return 1
    finally:
        if window_created:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
