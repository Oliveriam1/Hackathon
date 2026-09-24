"""Živý náhled kamery. Ukončení klávesou Q, Escape nebo zavřením okna."""

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.manual_target import ManualTarget
from src.vision import Vision, annotate_observation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    source.add_argument("--image", type=Path, help="Obrázek místo živé kamery.")
    source.add_argument("--demo", action="store_true", help="Testovací obraz bez kamery.")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")

    window_name = "Kamera - Q / Esc: konec"
    try:
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
                camera = stack.enter_context(Camera(args.camera))
                frame = camera.read()
            target = ManualTarget()
            vision = Vision()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, target.on_mouse)
            print("Levý klik: označit bod. Pravý klik: zrušit. R: reset reference. Q / Escape: konec.")
            while True:
                observation = vision.observe(frame)
                cv2.imshow(window_name, annotate_observation(target.annotate(frame), observation))
                key = cv2.waitKey(20) & 0xFF
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
    except (RuntimeError, OSError, cv2.error) as error:
        print(f"Chyba náhledu: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
