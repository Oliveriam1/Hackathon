"""Živý náhled kamery. Ukončení klávesou Q, Escape nebo zavřením okna."""

import argparse
import sys

import cv2

from src.camera import Camera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")

    window_name = "Kamera - Q / Esc: konec"
    try:
        with Camera(args.camera) as camera:
            frame = camera.read()
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            print("Náhled běží. Klikněte do okna a stiskněte Q nebo Escape pro ukončení.")
            while True:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                frame = camera.read()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, cv2.error) as error:
        print(f"Chyba kamery: {error}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
