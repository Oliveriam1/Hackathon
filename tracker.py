#!/usr/bin/env python3
"""
tracker.py - find the red (pink on NoIR) circle, show a live preview in a web
browser and send the angle over Wi-Fi.

Run on the Pi:   python3 tracker.py
Live preview:    http://<pi-address>:8000
Telemetry:       JSON over UDP port 5005 (see receive.py)

The camera model and full sensor resolution are read from the camera.
Field of view values are the official Raspberry Pi figures for each module.
"""

import json
import math
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

WEB_PORT = 8000
UDP_PORT = 5005
UDP_TARGET = "255.255.255.255"   # broadcast on the Wi-Fi, or put your PC's IP here
PREVIEW_WIDTH = 820              # only the size of the browser preview image
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# Official field of view (horizontal, vertical) in degrees at full sensor resolution
FOV = {
    "ov5647": (53.50, 41.41),       # Camera Module 1
    "imx219": (62.2, 48.8),         # Camera Module 2
    "imx708": (66.0, 41.0),         # Camera Module 3
    "imx708_wide": (102.0, 67.0),   # Camera Module 3 Wide
}

# Colour range measured from your photo (circle: H ~160, S ~35 on the NoIR camera).
# Adjust with the sliders on the web page; changes are saved to settings.json.
# min_area and min_roundness are 0 = off. Raise them only if you get false detections.
settings = {
    "h_min": 148, "h_max": 175,
    "s_min": 26, "s_max": 255,
    "v_min": 90, "v_max": 255,
    "min_area": 0,
    "min_roundness": 0,
    "show_mask": 0,
}
if os.path.exists(SETTINGS_FILE):
    try:
        settings.update(json.load(open(SETTINGS_FILE)))
    except Exception:
        pass


def make_mask(frame):
    s = settings
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo_sv, hi_sv = (s["s_min"], s["v_min"]), (s["s_max"], s["v_max"])
    if s["h_min"] <= s["h_max"]:
        return cv2.inRange(hsv, (s["h_min"], *lo_sv), (s["h_max"], *hi_sv))
    # range wraps through 0 (e.g. 170..10)
    return cv2.inRange(hsv, (s["h_min"], *lo_sv), (180, *hi_sv)) | \
        cv2.inRange(hsv, (0, *lo_sv), (s["h_max"], *hi_sv))


def find_circle(mask):
    """Pick the blob that is both largest and roundest. Returns dict or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_score = None, 0.0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.countNonZero(mask[y:y + h, x:x + w])
        _, r = cv2.minEnclosingCircle(c)
        # roundness: 1.0 = perfect filled circle, lower = line / irregular shape
        roundness = min(1.0, area / (math.pi * max(r, 0.5) ** 2))
        if area < settings["min_area"] or roundness * 100 < settings["min_roundness"]:
            continue
        score = area * roundness ** 2
        if score > best_score:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = x + w / 2, y + h / 2
            best_score = score
            best = {"x": cx, "y": cy, "r": r, "area": area, "roundness": roundness}
    return best


# --------------------------------------------------------------- web server --
latest_jpeg = None
latest_data = {"found": False}
frame_ready = threading.Condition()

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>Circle tracker</title><style>
body{font-family:sans-serif;background:#111;color:#eee;margin:10px}
img{max-width:100%;border:1px solid #444} #t{font-size:22px;margin:8px 0}
label{display:inline-block;width:110px} input[type=range]{width:220px}</style></head><body>
<img src="/stream"><div id=t>...</div><div id=i></div><div id=s></div>
<script>
const names=[["h_min",0,180],["h_max",0,180],["s_min",0,255],["s_max",0,255],["v_min",0,255],
 ["v_max",0,255],["min_area",0,500],["min_roundness",0,100],["show_mask",0,1]];
fetch('/settings').then(r=>r.json()).then(s=>{let h='';for(const [n,a,b] of names)
 h+=`<div><label>${n}</label><input type=range min=${a} max=${b} value=${s[n]}
 oninput="this.nextSibling.textContent=this.value;fetch('/set?${n}='+this.value)"><span>${s[n]}</span></div>`;
 document.getElementById('s').innerHTML=h;});
fetch('/info').then(r=>r.json()).then(d=>document.getElementById('i').textContent=
 `camera ${d.model}  ${d.width}x${d.height}  FOV ${d.hfov}° x ${d.vfov}°`);
setInterval(()=>fetch('/data').then(r=>r.json()).then(d=>{
 document.getElementById('t').textContent=d.found?
 `right ${d.right_deg.toFixed(2)}°  forward ${d.fwd_deg.toFixed(2)}°  total ${d.total_deg.toFixed(2)}°  | ${d.size_px} px, round ${d.roundness} | ${d.fps} fps`:
 `no target | ${d.fps} fps`;}),200);
</script></body></html>"""

camera_info = {}


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
        elif url.path == "/info":
            self.send(json.dumps(camera_info).encode(), "application/json")
        elif url.path == "/settings":
            self.send(json.dumps(settings).encode(), "application/json")
        elif url.path == "/set":
            for k, v in parse_qs(url.query).items():
                if k in settings:
                    settings[k] = int(v[0])
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
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


# --------------------------------------------------------------------- main --
def main():
    global latest_jpeg, latest_data
    from picamera2 import Picamera2

    cams = Picamera2.global_camera_info()
    if not cams:
        sys.exit("No camera found.")
    model = cams[0]["Model"]                    # e.g. imx219, imx708_wide_noir
    base = model.replace("_noir", "")
    if base not in FOV:
        sys.exit(f"Unknown camera model '{model}'. Add its field of view to FOV in tracker.py.")
    hfov, vfov = FOV[base]

    tuning_name = model if model.endswith("_noir") else model + "_noir"
    try:
        cam = Picamera2(tuning=Picamera2.load_tuning_file(tuning_name + ".json"))
    except Exception:
        print(f"NoIR tuning file {tuning_name}.json not found, using default tuning")
        cam = Picamera2()

    res = cam.sensor_resolution                  # full sensor = full field of view
    cam.configure(cam.create_video_configuration(
        main={"size": res, "format": "RGB888"}, buffer_count=2))
    cam.start()

    W, H = res
    fx = (W / 2) / math.tan(math.radians(hfov) / 2)
    fy = (H / 2) / math.tan(math.radians(vfov) / 2)
    camera_info.update(model=model, width=W, height=H, hfov=hfov, vfov=vfov)
    print(f"Camera {model}, {W}x{H}, FOV {hfov} x {vfov} deg")

    server = ThreadingHTTPServer(("", WEB_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"Preview: http://{socket.gethostname()}.local:{WEB_PORT}   Telemetry: UDP port {UDP_PORT}")

    fps, t_prev = 0.0, time.time()
    while True:
        frame = cam.capture_array("main")
        mask = make_mask(frame)
        hit = find_circle(mask)

        now = time.time()
        fps = 1.0 / max(now - t_prev, 1e-6) if fps == 0 else 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        data = {"t": round(now, 3), "found": hit is not None, "fps": round(fps, 1)}
        if hit:
            tx = (hit["x"] - (W - 1) / 2) / fx      # + = right in image
            ty = ((H - 1) / 2 - hit["y"]) / fy      # + = top of image
            data.update(
                right_deg=round(math.degrees(math.atan(tx)), 3),
                fwd_deg=round(math.degrees(math.atan(ty)), 3),
                total_deg=round(math.degrees(math.atan(math.hypot(tx, ty))), 3),
                x=round(hit["x"], 1), y=round(hit["y"], 1),
                size_px=round(2 * hit["r"], 1), roundness=round(hit["roundness"], 2))
        latest_data = data
        try:
            udp.sendto(json.dumps(data).encode(), (UDP_TARGET, UDP_PORT))
        except OSError:
            pass

        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if settings["show_mask"] else frame.copy()
        k = W / PREVIEW_WIDTH                         # keep markings visible after downscaling
        cv2.drawMarker(view, (W // 2, H // 2), (255, 255, 0), cv2.MARKER_CROSS, int(30 * k), int(2 * k))
        if hit:
            c = (int(hit["x"]), int(hit["y"]))
            cv2.circle(view, c, int(hit["r"] + 10 * k), (0, 255, 0), int(2 * k))
            cv2.line(view, (W // 2, H // 2), c, (0, 255, 0), int(k))
        small = cv2.resize(view, (PREVIEW_WIDTH, int(H / k)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_ready:
            latest_jpeg = jpg.tobytes()
            frame_ready.notify_all()


if __name__ == "__main__":
    main()
