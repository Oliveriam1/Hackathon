#!/usr/bin/env python3
"""
tracker.py - find a red (pink on NoIR) circle, show a live preview in a web
browser and send the angle over Wi-Fi.

Run on the Pi:      python3 tracker.py
Live preview:       open  http://<pi-address>:8000  in a browser
Telemetry:          JSON over UDP to port 5005 (see receive.py)
"""

import json
import math
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

# ----------------------------------------------------------------- settings --
CAMERA_RES = (1640, 1232)   # full field of view on camera v2
HFOV_DEG = 62.2             # camera v2. v1: 53.5 / 41.41   v3: 66 / 41
VFOV_DEG = 48.8
WEB_PORT = 8000
UDP_PORT = 5005
UDP_TARGET = "255.255.255.255"   # broadcast to everyone on the Wi-Fi, or put your laptop's IP
PREVIEW_WIDTH = 820              # size of the browser preview
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# Colour defaults measured from your photo (circle looked pink: H~160, S~35).
# Change them live with the sliders on the web page; they are saved automatically.
settings = {
    "h_min": 148, "h_max": 175,   # hue range (0-180). if h_min > h_max it wraps through 0
    "s_min": 26, "s_max": 255,
    "v_min": 90, "v_max": 255,
    "min_area": 40,               # pixels. 200 mm circle at 20 m is ~145 px
    "min_fill": 0.6,              # how round/filled the blob must be (0-1)
    "show_mask": 0,
}
if os.path.exists(SETTINGS_FILE):
    try:
        settings.update(json.load(open(SETTINGS_FILE)))
    except Exception:
        pass

# ------------------------------------------------------------------ helpers --
fx = (CAMERA_RES[0] / 2) / math.tan(math.radians(HFOV_DEG) / 2)
fy = (CAMERA_RES[1] / 2) / math.tan(math.radians(VFOV_DEG) / 2)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def make_mask(frame):
    s = settings
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo_sv, hi_sv = (s["s_min"], s["v_min"]), (s["s_max"], s["v_max"])
    if s["h_min"] <= s["h_max"]:
        m = cv2.inRange(hsv, (s["h_min"], *lo_sv), (s["h_max"], *hi_sv))
    else:  # wraps around red (e.g. 170..10)
        m = cv2.inRange(hsv, (s["h_min"], *lo_sv), (180, *hi_sv)) | \
            cv2.inRange(hsv, (0, *lo_sv), (s["h_max"], *hi_sv))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def find_circle(mask, last):
    """Return (x, y, radius) of the best round blob, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.countNonZero(mask[y:y + h, x:x + w])
        if area < settings["min_area"] or not (0.6 < w / h < 1.66):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = area / (math.pi * r * r)
        if fill < settings["min_fill"]:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        score = fill * math.sqrt(area)
        if last:  # prefer the blob near where we saw it last frame
            score /= 1 + math.hypot(cx - last[0], cy - last[1]) / 200
        if score > best_score:
            best, best_score = (cx, cy, r), score
    return best


def angles(x, y, w, h):
    tx = (x - (w - 1) / 2) / (fx * w / CAMERA_RES[0])
    ty = ((h - 1) / 2 - y) / (fy * h / CAMERA_RES[1])
    return (math.degrees(math.atan(tx)), math.degrees(math.atan(ty)),
            math.degrees(math.atan(math.hypot(tx, ty))))


# --------------------------------------------------------------- web server --
latest_jpeg = None
latest_data = {"found": False}
frame_ready = threading.Condition()

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>Circle tracker</title><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:10px}
img{max-width:100%;border:1px solid #444} #t{font-size:22px;margin:8px 0}
label{display:inline-block;width:80px} input[type=range]{width:220px}
.row{margin:4px 0}</style></head><body>
<img src="/stream"><div id=t>...</div><div id=s></div>
<script>
const names=[["h_min",0,180],["h_max",0,180],["s_min",0,255],["s_max",0,255],
 ["v_min",0,255],["v_max",0,255],["min_area",5,2000],["min_fill",0,100],["show_mask",0,1]];
fetch('/settings').then(r=>r.json()).then(s=>{
 let h='';for(const [n,a,b] of names){let v=n=='min_fill'?Math.round(s[n]*100):s[n];
 h+=`<div class=row><label>${n}</label><input type=range min=${a} max=${b} value=${v}
 oninput="this.nextSibling.textContent=this.value;fetch('/set?${n}='+this.value)"><span>${v}</span></div>`}
 document.getElementById('s').innerHTML=h;});
setInterval(()=>fetch('/data').then(r=>r.json()).then(d=>{
 document.getElementById('t').textContent=d.found?
 `right ${d.right_deg.toFixed(2)}°   forward ${d.fwd_deg.toFixed(2)}°   total ${d.total_deg.toFixed(2)}°   (${d.fps} fps)`:
 `no target   (${d.fps} fps)`;}),200);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send(PAGE.encode(), "text/html")
        elif url.path == "/data":
            self.send(json.dumps(latest_data).encode(), "application/json")
        elif url.path == "/settings":
            self.send(json.dumps(settings).encode(), "application/json")
        elif url.path == "/set":
            for k, v in parse_qs(url.query).items():
                if k in settings:
                    settings[k] = float(v[0]) / 100 if k == "min_fill" else int(v[0])
            json.dump(settings, open(SETTINGS_FILE, "w"))
            self.send(b"ok", "text/plain")
        elif url.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_ready:
                        frame_ready.wait(timeout=2)
                        jpg = latest_jpeg
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


# --------------------------------------------------------------------- main --
def main():
    global latest_jpeg, latest_data
    from picamera2 import Picamera2

    try:
        cam = Picamera2(tuning=Picamera2.load_tuning_file("imx219_noir.json"))
    except Exception:
        cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": CAMERA_RES, "format": "RGB888"}, buffer_count=2))
    cam.start()
    cam.set_controls({"ExposureTime": 3000, "AnalogueGain": 2.0})  # short exposure = less blur
    time.sleep(1)

    server = ThreadingHTTPServer(("", WEB_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"Preview: http://{socket.gethostname()}.local:{WEB_PORT}   Telemetry: UDP port {UDP_PORT}")

    last, fps, t_prev = None, 0.0, time.time()
    while True:
        frame = cam.capture_array("main")
        h, w = frame.shape[:2]
        mask = make_mask(frame)
        hit = find_circle(mask, last)
        last = hit[:2] if hit else None

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        data = {"t": round(now, 3), "found": hit is not None, "fps": round(fps, 1)}
        if hit:
            r_deg, f_deg, tot = angles(hit[0], hit[1], w, h)
            data.update(right_deg=round(r_deg, 3), fwd_deg=round(f_deg, 3),
                        total_deg=round(tot, 3), x=round(hit[0], 1), y=round(hit[1], 1))
        latest_data = data
        try:
            udp.sendto(json.dumps(data).encode(), (UDP_TARGET, UDP_PORT))
        except OSError:
            pass  # no network right now

        # Preview image
        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if settings["show_mask"] else frame.copy()
        cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 60, 3)
        if hit:
            c = (int(hit[0]), int(hit[1]))
            cv2.circle(view, c, int(hit[2]) + 15, (0, 255, 0), 4)
            cv2.line(view, (w // 2, h // 2), c, (0, 255, 0), 2)
        small = cv2.resize(view, (PREVIEW_WIDTH, int(h * PREVIEW_WIDTH / w)))
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_ready:
            latest_jpeg = jpg.tobytes()
            frame_ready.notify_all()


if __name__ == "__main__":
    main()
