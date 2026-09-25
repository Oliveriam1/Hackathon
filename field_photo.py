#!/usr/bin/env python3
"""Photograph the field every 10 seconds and show the latest photo in a browser.

Place the UAV above the field centre, point the camera vertically down and align
the field width with the image width. Assumes flat ground. Height is AGL, not GNSS
altitude. Default field 60x30 m and nominal FOV 54x41 degrees follow red_tracker.py;
measure the actual lens FOV before relying on the coverage estimate.

This script does not fly the UAV or move servos. --plan-only needs no camera.
The saved image has no annotations, resizing, detection or colour correction.
Run: python3 field_photo.py
On a PC on the same network open http://RASPBERRY_PI_IP:8000/.
No receiver program is needed on the PC. Photos are also saved locally.
The web root is ./web_photos by default (--web-root changes it). Timestamped
photos and latest.png (or latest.jpg) are saved there and available from /.
Optional --send-url retains HTTP POST upload support. Ctrl+C closes everything.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Callable
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen

from src.pi_camera import PiCamera


PHOTO_PAGE = """<!doctype html>
<html lang="cs"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fotky pole</title>
<style>
body{background:#16191d;color:#eee;font:16px sans-serif;margin:20px}
img{display:block;max-width:100%;height:auto;margin-top:16px}
a{color:#8fcfff;margin-left:20px} label{display:inline-block;margin:12px 0}
</style>
<h1>Fotky pole</h1>
<p id="status">Čekám na první fotku…</p>
<label><input id="automatic" type="checkbox" checked> Automaticky zobrazovat nové fotky</label>
<a id="download" hidden>Stáhnout zobrazenou fotku</a>
<img id="photo" alt="Fotka pole" hidden>
<script>
let etag = '', previousURL = null;
async function update() {
  if (!document.getElementById('automatic').checked) {
    setTimeout(update, 1000); return;
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch('/latest', {
      cache: 'no-store', signal: controller.signal,
      headers: etag ? {'If-None-Match': etag} : {}
    });
    if (response.status === 503) {
      document.getElementById('status').textContent = 'Čekám na první fotku…';
    } else if (response.status === 304) {
      document.getElementById('status').textContent = 'Zobrazeno: ' + document.getElementById('download').download;
    } else {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const blob = await response.blob();
      const name = decodeURIComponent(response.headers.get('X-Photo-Filename'));
      const nextURL = URL.createObjectURL(blob);
      const image = document.getElementById('photo');
      const download = document.getElementById('download');
      image.src = nextURL; image.hidden = false;
      download.href = nextURL; download.download = name; download.hidden = false;
      if (previousURL) URL.revokeObjectURL(previousURL);
      previousURL = nextURL;
      etag = response.headers.get('ETag');
      document.getElementById('status').textContent = 'Zobrazeno: ' + name;
    }
  } catch (error) {
    document.getElementById('status').textContent = 'Spojení přerušeno — zkouším znovu. Zobrazená fotka může být stará.';
  } finally {
    clearTimeout(timeout);
    setTimeout(update, 1000);
  }
}
update();
</script></html>"""


class PhotoServer:
    """Serve the viewer and photographs directly from the configured web root.

    Local-network viewer, without authentication. No directory listing or source files.
    The camera publishes a complete image under a short lock after saving it.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8000,
                 root: Path = Path("web_photos")) -> None:
        self.host, self.port = host, port
        self.root = root.resolve()
        self._lock = threading.Lock()
        self._latest: tuple[bytes, str, str, str] | None = None
        self._version = 0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(self, path: Path) -> None:
        if path.resolve().parent != self.root:
            raise ValueError("Fotka musi byt ulozena primo v korenove slozce webu")
        data = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        # Atomic replacement prevents a browser from reading a half-written photo.
        latest_path = self.root / ("latest.png" if mime == "image/png" else "latest.jpg")
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".photo-", delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
            os.replace(temp_path, latest_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        with self._lock:
            self._version += 1
            self._latest = data, mime, path.name, f'"{self._version}"'

    def __enter__(self) -> PhotoServer:
        owner = self
        self.root.mkdir(parents=True, exist_ok=True)

        class Handler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                self.request.settimeout(5)
                super().setup()

            def do_GET(self) -> None:
                try:
                    path = urlsplit(self.path).path
                    if path == "/":
                        data, mime = PHOTO_PAGE.encode("utf-8"), "text/html; charset=utf-8"
                        headers = {}
                    elif path == "/latest":
                        with owner._lock:
                            latest = owner._latest
                        if latest is None:
                            self.send_error(503, "Waiting for first photo")
                            return
                        data, mime, name, etag = latest
                        if self.headers.get("If-None-Match") == etag:
                            self.send_response(304)
                            self.send_header("ETag", etag)
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            return
                        headers = {"ETag": etag, "X-Photo-Filename": quote(name, safe="")}
                    else:
                        # Only top-level image files, never arbitrary project paths.
                        name = unquote(path.lstrip("/"))
                        image_path = (owner.root / name).resolve()
                        if (not name or Path(name).name != name or name.startswith(".")
                                or image_path.parent != owner.root
                                or image_path.suffix.lower() not in (".png", ".jpg", ".jpeg")
                                or not image_path.is_file()):
                            self.send_error(404)
                            return
                        data = image_path.read_bytes()
                        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
                        headers = {"X-Photo-Filename": quote(name, safe="")}
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # A disconnected browser must not stop photography.

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        print(f"Fotky na PC: otevri http://IP_RASPBERRY:{self.port}/ ve stejne siti.", flush=True)
        print(f"Korenova slozka webu: {self.root}", flush=True)
        return self

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


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


def photograph_periodically(output: Path, *, interval: float = 5.0, delay: float = 0.0,
                           send_url: str | None = None, count: int = 0,
                           on_photo: Callable[[Path], None] | None = None) -> None:
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
    elif on_photo is None:
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
                if on_photo is not None:
                    on_photo(path)
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
    parser.add_argument("--interval", type=float, default=5, help="Interval snimku v sekundach (default: 10)")
    parser.add_argument("--count", type=int, default=0, help="Pocet fotek; 0 = az do Ctrl+C")
    parser.add_argument("--send-url", help="HTTP endpoint prijimajici POST s obrazkem")
    parser.add_argument("--host", default="0.0.0.0", help="Adresa webu; default: vsechna sitova rozhrani")
    parser.add_argument("--port", type=int, default=8000, help="Port pro fotky v prohlizeci (default: 8000)")
    parser.add_argument("--web-root", type=Path, default=Path("web_photos"),
                        help="Slozka, do ktere web primo uklada a ze ktere poskytuje fotky")
    parser.add_argument("--output", type=Path, default=Path("pole.png"),
                        help="Zaklad nazvu fotek uvnitr --web-root; prida se cas a poradove cislo")
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
        output = (args.web_root / args.output).resolve()
        if output.parent != args.web_root.resolve():
            raise ValueError("--output musi lezet primo v --web-root; slozku nastav pomoci --web-root")
        with PhotoServer(args.host, args.port, args.web_root) as viewer:
            photograph_periodically(output, interval=args.interval, delay=args.delay,
                                   send_url=args.send_url, count=args.count, on_photo=viewer.publish)
            if args.count:
                print("Foceni dokonceno. Web zustava dostupny do Ctrl+C.", flush=True)
                while True:
                    time.sleep(1)
        return 0
    except KeyboardInterrupt:
        print("Foceni zruseno.")
        return 130
    except Exception as exc:
        print(f"CHYBA: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
