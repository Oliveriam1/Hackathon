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
from src.vision import Vision, annotate_observation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    source.add_argument("--image", type=Path, help="Obrázek místo živé kamery.")
    source.add_argument("--demo", action="store_true", help="Testovací obraz bez kamery.")
    source.add_argument("--picamera", type=int, metavar="INDEX", help="CSI kamera přes Picamera2, např. --picamera 0.")
    parser.add_argument("--snapshot", type=Path, help="Uloží jeden snímek bez grafického okna (např. test.jpg).")
    parser.add_argument('--ev', type=float, default=None,
                        help='Kompenzace expozice CSI kamery, např. --ev -2. Výchozí 0; AWB auto.')
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")
    if args.picamera is not None and args.picamera < 0:
        parser.error("Index CSI kamery musí být nezáporný.")
    if args.ev is not None and args.picamera is None:
        parser.error('--ev lze použít pouze s --picamera.')
    if args.ev is not None and (not np.isfinite(args.ev) or not -8 <= args.ev <= 8):
        parser.error('--ev musí být v rozsahu -8 až 8.')

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
                device = PiCamera(args.picamera, ev=args.ev if args.ev is not None else 0.0) if args.picamera is not None else Camera(args.camera)
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
            vision = Vision()
            show_edges = False
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
            print("Detekce koleček. 1/2/3: citlivost, E: hrany, Q / Escape: konec.")
            while True:
                observation = vision.observe(frame)
                background = cv2.cvtColor(vision.detector.edges, cv2.COLOR_GRAY2BGR) if show_edges else frame
                image = annotate_observation(background, observation)
                cv2.imshow(window_name, image)
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('1'), ord('2'), ord('3')):
                    vision.detector.sensitivity = int(chr(key))
                if key in (ord('e'), ord('E')):
                    show_edges = not show_edges
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
