#!/usr/bin/env python3
"""Estimate field coverage and save one clean camera photo for manual mapping.

Place the UAV above the field centre, point the camera vertically down and align
the field width with the image width. Assumes flat ground. Height is AGL, not GNSS
altitude. Default field 60x30 m and nominal FOV 54x41 degrees follow red_tracker.py;
measure the actual lens FOV before relying on the coverage estimate.

This script does not fly the UAV or move servos. --plan-only needs no camera.
The saved image has no annotations, resizing, detection or colour correction.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import time

from src.pi_camera import PiCamera


def required_height(field_width: float, field_height: float, hfov: float,
                    vfov: float, margin: float = 1.1) -> float:
    """Minimum estimated AGL for a centred, level camera, with linear margin."""
    if not all(math.isfinite(v) for v in (field_width, field_height, hfov, vfov, margin)):
        raise ValueError("Rozmery, FOV a rezerva musi byt konecna cisla")
    if field_width <= 0 or field_height <= 0:
        raise ValueError("Rozmery pole musi byt kladne")
    if not (0 < hfov < 180 and 0 < vfov < 180):
        raise ValueError("FOV musi byt mezi 0 a 180 stupni")
    if margin < 1:
        raise ValueError("Rezerva musi byt alespon 1.0")
    height = margin * max(field_width / (2 * math.tan(math.radians(hfov / 2))),
                          field_height / (2 * math.tan(math.radians(vfov / 2))))
    if not math.isfinite(height):
        raise ValueError("Vysku pro zadane hodnoty nelze vypocitat")
    return height


def take_photo(output: Path, delay: float = 3.0) -> tuple[int, int]:
    """Capture BGR via Picamera2 at the tracker's 4:3 resolution; never overwrite."""
    import cv2

    if not math.isfinite(delay) or delay < 0:
        raise ValueError("Odpocet musi byt nezaporny")
    suffix = output.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Vystup musi mit priponu .png nebo .jpg")
    if output.exists():
        raise FileExistsError(f"Soubor uz existuje: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"Vystupni slozka neexistuje: {output.parent}")
    with PiCamera(width=1296, height=972) as camera:
        print(f"Kamera pripravena. Fotim za {delay:g} s...", flush=True)
        time.sleep(delay)
        frame = camera.read()
    options = [cv2.IMWRITE_JPEG_QUALITY, 100] if suffix in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(suffix, frame, options)
    if not ok:
        raise RuntimeError("Fotografii se nepodarilo zakodovat")
    with output.open("xb") as handle:
        handle.write(encoded.tobytes())
    return frame.shape[1], frame.shape[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-width", type=float, default=60, help="Sirka pole v metrech (default: 60)")
    parser.add_argument("--field-height", type=float, default=30, help="Delka pole v metrech (default: 30)")
    parser.add_argument("--hfov", type=float, default=54, help="Horizontalni FOV ve stupnich (nominalne 54)")
    parser.add_argument("--vfov", type=float, default=41, help="Vertikalni FOV ve stupnich (nominalne 41)")
    parser.add_argument("--margin", type=float, default=1.1, help="Rezerva pokryti (default: 1.1 = 10 %%)")
    parser.add_argument("--delay", type=float, default=3, help="Sekundy do foceni po otevreni kamery")
    parser.add_argument("--output", type=Path,
                        default=Path(datetime.now().strftime("pole_%Y%m%d_%H%M%S_%f.png")))
    parser.add_argument("--plan-only", action="store_true", help="Pouze vypocitat vysku, neotevirat kameru")
    args = parser.parse_args()
    try:
        height = required_height(args.field_width, args.field_height, args.hfov, args.vfov, args.margin)
        print(f"Pole: {args.field_width:g} x {args.field_height:g} m")
        print(f"Odhad potrebne vysky: {height:.2f} m NAD ZEMI (AGL)")
        print("Pred focenim umisti dron nad stred pole a kameru kolmo dolu.")
        print("Sirka pole musi smerovat podel sirky obrazu. FOV je nutne overit pro konkretni kameru.")
        if args.plan_only:
            return 0
        width, height_px = take_photo(args.output, args.delay)
        print(f"Ulozeno: {args.output.resolve()} ({width} x {height_px} px)")
        return 0
    except KeyboardInterrupt:
        print("Foceni zruseno.")
        return 130
    except Exception as exc:
        print(f"CHYBA: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
