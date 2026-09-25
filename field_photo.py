#!/usr/bin/env python3
"""Photograph the field every 10 seconds; optionally upload each image over HTTP.

Place the UAV above the field centre, point the camera vertically down and align
the field width with the image width. Assumes flat ground. Height is AGL, not GNSS
altitude. Default field 60x30 m and nominal FOV 54x41 degrees follow red_tracker.py;
measure the actual lens FOV before relying on the coverage estimate.

This script does not fly the UAV or move servos. --plan-only needs no camera.
The saved image has no annotations, resizing, detection or colour correction.
--send-url must point to a receiver accepting an HTTP POST with raw image bytes.
Without --send-url, photos are saved locally only. Ctrl+C closes the camera.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

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


def send_photo(output: Path, url: str, timeout: float = 3.0) -> None:
    """POST raw PNG/JPEG bytes. The receiver must return a successful 2xx status.

    This is not multipart/form-data. X-Photo-Filename identifies the local file.
    A failed upload leaves the local photograph available for later retrieval.
    """
    content_type = "image/png" if output.suffix.lower() == ".png" else "image/jpeg"
    request = Request(url, data=output.read_bytes(), method="POST", headers={
        "Content-Type": content_type,
        "X-Photo-Filename": output.name.encode("ascii", errors="replace").decode("ascii"),
    })
    with urlopen(request, timeout=timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Prijemce vratil HTTP {response.status}")


def photograph_periodically(output: Path, *, interval: float = 10.0, delay: float = 0.0,
                           send_url: str | None = None, count: int = 0) -> None:
    """Keep one camera open; save unique frames on a monotonic schedule.

    count=0 runs until Ctrl+C. Slow cycles skip missed deadlines, rather than
    producing a burst of old photos. Upload failures do not stop acquisition.
    """
    import cv2

    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Interval musi byt kladny")
    if not math.isfinite(delay) or delay < 0 or count < 0:
        raise ValueError("Odpocet a pocet snimku musi byt nezaporny")
    suffix = output.suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Vystup musi mit priponu .png nebo .jpg")
    if not output.parent.is_dir():
        raise ValueError(f"Vystupni slozka neexistuje: {output.parent}")
    if send_url:
        parsed = urlsplit(send_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Cil odesilani musi byt platna http:// nebo https:// URL")
    else:
        print("Odesilani neni nastavene: snimky se pouze ukladaji. Pro odesilani pouzij --send-url.", flush=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, 100] if suffix in (".jpg", ".jpeg") else []
    with PiCamera(width=1296, height=972) as camera:
        print(f"Kamera pripravena. Snimek kazdych {interval:g} s; ukonceni Ctrl+C.", flush=True)
        deadline = time.monotonic() + delay
        captured = 0
        while count == 0 or captured < count:
            time.sleep(max(0.0, deadline - time.monotonic()))
            try:
                frame = camera.read()
                ok, encoded = cv2.imencode(suffix, frame, options)
                if not ok:
                    raise RuntimeError("Fotografii se nepodarilo zakodovat")
            except (ValueError, RuntimeError, cv2.error) as exc:
                print(f"CHYBA snimku: {exc}; zkusim dalsi interval.", flush=True)
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = output.with_name(f"{output.stem}_{stamp}_{captured + 1:06d}{suffix}")
                with path.open("xb") as handle:
                    handle.write(encoded.tobytes())
                captured += 1
                print(f"Ulozeno: {path.resolve()}", flush=True)
                if send_url:
                    try:
                        send_photo(path, send_url, timeout=min(3.0, interval / 2))
                        print(f"Odeslano: {path.name}", flush=True)
                    except Exception as exc:
                        print(f"ODESLANI SELHALO: {exc}. Snimek zustava ulozeny: {path}", flush=True)
            deadline += interval
            now = time.monotonic()
            if deadline < now:
                deadline += (math.floor((now - deadline) / interval) + 1) * interval


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-width", type=float, default=60, help="Sirka pole v metrech (default: 60)")
    parser.add_argument("--field-height", type=float, default=30, help="Delka pole v metrech (default: 30)")
    parser.add_argument("--hfov", type=float, default=54, help="Horizontalni FOV ve stupnich (nominalne 54)")
    parser.add_argument("--vfov", type=float, default=41, help="Vertikalni FOV ve stupnich (nominalne 41)")
    parser.add_argument("--margin", type=float, default=1.1, help="Rezerva pokryti (default: 1.1 = 10 %%)")
    parser.add_argument("--delay", type=float, default=0, help="Sekundy do prvni fotky po otevreni kamery")
    parser.add_argument("--interval", type=float, default=10, help="Interval snimku v sekundach (default: 10)")
    parser.add_argument("--count", type=int, default=0, help="Pocet fotek; 0 = az do Ctrl+C")
    parser.add_argument("--send-url", help="HTTP endpoint prijimajici POST s obrazkem")
    parser.add_argument("--output", type=Path, default=Path("pole.png"),
                        help="Zaklad nazvu fotek; ke kazde se prida cas a poradove cislo")
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
        photograph_periodically(args.output, interval=args.interval, delay=args.delay,
                               send_url=args.send_url, count=args.count)
        return 0
    except KeyboardInterrupt:
        print("Foceni zruseno.")
        return 130
    except Exception as exc:
        print(f"CHYBA: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
