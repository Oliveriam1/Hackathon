#!/usr/bin/env python3
"""
red_circle_tracker.py - detect and track a red circular target from a
downward-looking Raspberry Pi camera and report its angle from the optical axis.

Target:  red circle, 200 mm diameter, on concrete
Camera:  Raspberry Pi NoIR camera (v1 / v2 / v3), pointing straight down
Host:    Raspberry Pi 3A+ (Raspberry Pi OS Bookworm/Bullseye, picamera2)

Output (one JSON line per frame on stdout), e.g.
{"t": 12.34, "found": true, "u": 901.2, "v": 544.8, "diam_px": 13.4,
 "angle_right_deg": 2.61, "angle_fwd_deg": 1.13, "off_nadir_deg": 2.84,
 "bearing_deg": 66.6, "ground_right_m": 0.91, "ground_fwd_m": 0.39,
 "range_from_size_m": 20.3, "tracking": true, "fps": 7.8}

Conventions (camera looking straight down, top of the image = drone nose):
  angle_right_deg  + means the target is to the RIGHT of centre
  angle_fwd_deg    + means the target is toward the TOP of the image (forward)
  off_nadir_deg    total angle between the optical axis and the target
  bearing_deg      direction to target, 0 = image top/forward, 90 = right (clockwise)
  ground_*_m       horizontal offset on the ground (needs --altitude)

Usage:
  python3 red_circle_tracker.py --camera v2 --altitude 20            # live camera
  python3 red_circle_tracker.py --image test.jpg --altitude 20 --show # test on a still
  python3 red_circle_tracker.py --save-raw 20                         # capture frames for HSV tuning
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

# Full-sensor field of view and a resolution that uses the whole sensor (binned).
# Using a cropped mode would change the FOV, so keep these unless you know better.
CAMERAS = {
    "v1":     dict(hfov=53.50, vfov=41.41, res=(1296, 972),  tuning="ov5647_noir.json"),
    "v2":     dict(hfov=62.20, vfov=48.80, res=(1640, 1232), tuning="imx219_noir.json"),
    "v3":     dict(hfov=66.00, vfov=41.00, res=(2304, 1296), tuning="imx708_noir.json"),
    "v3wide": dict(hfov=102.0, vfov=67.00, res=(2304, 1296), tuning="imx708_wide_noir.json"),
}

TARGET_DIAMETER_M = 0.200


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
class CameraModel:
    """Pinhole model: converts pixels to angles (and back)."""

    def __init__(self, width, height, hfov_deg, vfov_deg):
        self.w, self.h = width, height
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)

    def expected_diameter_px(self, altitude_m, diameter_m=TARGET_DIAMETER_M):
        return diameter_m * 0.5 * (self.fx + self.fy) / altitude_m

    def angles(self, u, v, altitude_m=None, diam_px=None):
        x = (u - self.cx) / self.fx          # tan(angle) to the right
        y = (self.cy - v) / self.fy          # tan(angle) forward (image up)
        out = {
            "angle_right_deg": math.degrees(math.atan(x)),
            "angle_fwd_deg": math.degrees(math.atan(y)),
            "off_nadir_deg": math.degrees(math.atan(math.hypot(x, y))),
            "bearing_deg": (math.degrees(math.atan2(x, y)) + 360.0) % 360.0,
        }
        if altitude_m:
            out["ground_right_m"] = altitude_m * x
            out["ground_fwd_m"] = altitude_m * y
        if diam_px and diam_px > 0:
            # Slant range estimated from apparent size (noisy when the target is small)
            f = 0.5 * (self.fx + self.fy)
            out["range_from_size_m"] = TARGET_DIAMETER_M * f / diam_px
        return out


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
class RedCircleDetector:
    def __init__(self, args):
        self.a = args
        k = max(1, args.morph)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def mask(self, bgr):
        a = self.a
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # Red wraps around hue 0/180 in OpenCV, so use two ranges.
        m1 = cv2.inRange(hsv, (0, a.s_min, a.v_min), (a.h_low, 255, 255))
        m2 = cv2.inRange(hsv, (a.h_high, a.s_min, a.v_min), (180, 255, 255))
        m = cv2.bitwise_or(m1, m2)
        if a.morph > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
        return m

    def detect(self, bgr, expected_d_px=None, offset=(0, 0), predict=None):
        """Return a list of candidates sorted best first.

        expected_d_px: expected target diameter in pixels (from altitude), or None
        offset:        (x0, y0) of this crop inside the full frame
        predict:       predicted (u, v) in full-frame coords, used to prefer nearby blobs
        """
        a = self.a
        m = self.mask(bgr)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if expected_d_px:
            exp_area = math.pi * (expected_d_px / 2.0) ** 2
            min_area = max(a.min_area, exp_area / a.size_tol ** 2)
            max_area = exp_area * a.size_tol ** 2
        else:
            exp_area = None
            min_area, max_area = a.min_area, a.max_area

        cands = []
        for c in contours:
            area = cv2.contourArea(c)
            npix = max(area, len(c))  # tiny blobs: contourArea underestimates
            if npix < min_area or npix > max_area:
                continue
            (xc, yc), r = cv2.minEnclosingCircle(c)
            if r < 1.5:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect < 1.0 / a.max_aspect or aspect > a.max_aspect:
                continue
            # How well does the blob fill its enclosing circle? (circle = ~1, line/L-shape = low)
            blob = m[y:y + h, x:x + w]
            filled = cv2.countNonZero(blob)
            fill = filled / (math.pi * r * r)
            if fill < a.min_fill:
                continue

            # Sub-pixel centroid from the mask pixels
            M = cv2.moments(blob, binaryImage=True)
            if M["m00"] == 0:
                continue
            u = x + M["m10"] / M["m00"] + offset[0]
            v = y + M["m01"] / M["m00"] + offset[1]
            diam = 2.0 * math.sqrt(filled / math.pi)  # equivalent diameter

            score = min(fill, 1.0)
            if exp_area:
                score *= math.exp(-abs(math.log(max(filled, 1) / exp_area)))
            if predict is not None:
                dist = math.hypot(u - predict[0], v - predict[1])
                score *= math.exp(-dist / max(a.track_radius, 1))
            cands.append(dict(u=u, v=v, diam_px=diam, fill=fill, score=score))

        cands.sort(key=lambda d: d["score"], reverse=True)
        return cands, m


# --------------------------------------------------------------------------- #
# Tracking (alpha-beta filter + region-of-interest search)
# --------------------------------------------------------------------------- #
class Tracker:
    def __init__(self, detector, cam, args):
        self.det, self.cam, self.a = detector, cam, args
        self.pos = None          # filtered (u, v)
        self.vel = (0.0, 0.0)    # px / frame
        self.misses = 0
        self.last_mask = None

    @property
    def tracking(self):
        return self.pos is not None

    def _roi(self, center, half):
        u, v = center
        x0 = int(max(0, u - half)); y0 = int(max(0, v - half))
        x1 = int(min(self.cam.w, u + half)); y1 = int(min(self.cam.h, v + half))
        return x0, y0, x1, y1

    def update(self, frame, expected_d_px):
        a = self.a
        predict = None
        best = None

        if self.pos is not None:
            predict = (self.pos[0] + self.vel[0], self.pos[1] + self.vel[1])
            half = max(a.track_radius, 6 * (expected_d_px or 10))
            x0, y0, x1, y1 = self._roi(predict, half)
            if x1 - x0 > 8 and y1 - y0 > 8:
                cands, self.last_mask = self.det.detect(
                    frame[y0:y1, x0:x1], expected_d_px, (x0, y0), predict)
                if cands:
                    best = cands[0]

        if best is None:  # full-frame search
            cands, self.last_mask = self.det.detect(frame, expected_d_px, (0, 0), predict)
            if cands:
                best = cands[0]

        if best is None:
            self.misses += 1
            if self.misses > a.max_misses:
                self.pos, self.vel = None, (0.0, 0.0)
            return None

        z = (best["u"], best["v"])
        if self.pos is None:
            self.pos, self.vel = z, (0.0, 0.0)
        else:
            al, be = a.alpha, a.beta
            px, py = predict
            rx, ry = z[0] - px, z[1] - py
            self.pos = (px + al * rx, py + al * ry)
            self.vel = (self.vel[0] + be * rx, self.vel[1] + be * ry)
        self.misses = 0
        best["u_filt"], best["v_filt"] = self.pos
        return best


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class PiCameraSource:
    def __init__(self, args, res, tuning_name):
        from picamera2 import Picamera2  # imported here so --image works on a PC
        tuning = None
        if not args.no_noir_tuning:
            try:
                tuning = Picamera2.load_tuning_file(args.tuning or tuning_name)
            except Exception as e:  # tuning file missing -> default tuning
                print(f"# tuning file not loaded ({e}); using default", file=sys.stderr)
        self.cam = Picamera2(tuning=tuning) if tuning else Picamera2()
        cfg = self.cam.create_video_configuration(
            main={"size": tuple(res), "format": "RGB888"},  # RGB888 == BGR order for OpenCV
            buffer_count=2)
        self.cam.configure(cfg)
        controls = {}
        if args.exposure_us:
            controls["ExposureTime"] = int(args.exposure_us)
            controls["AeEnable"] = False
            controls["AnalogueGain"] = float(args.gain)
        if args.awb_gains:
            controls["AwbEnable"] = False
            controls["ColourGains"] = tuple(args.awb_gains)
        if controls:
            self.cam.set_controls(controls)
        self.cam.start()
        time.sleep(1.0)  # let AE/AWB settle

    def read(self):
        return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()


class FileSource:
    def __init__(self, path, loop=False):
        self.imgs = None
        self.cap = None
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path)
                           if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
            self.imgs = [os.path.join(path, f) for f in files]
        elif path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            self.imgs = [path]
        else:
            self.cap = cv2.VideoCapture(path)
        self.i = 0

    def read(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            return f if ok else None
        if self.i >= len(self.imgs):
            return None
        f = cv2.imread(self.imgs[self.i]); self.i += 1
        return f

    def close(self):
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def annotate(frame, cam, det, info):
    out = frame.copy()
    cx, cy = int(cam.cx), int(cam.cy)
    cv2.drawMarker(out, (cx, cy), (255, 255, 0), cv2.MARKER_CROSS, 40, 2)
    if det:
        u, v = int(det["u"]), int(det["v"])
        r = int(max(det["diam_px"], 6))
        cv2.circle(out, (u, v), r + 6, (0, 255, 0), 2)
        cv2.line(out, (cx, cy), (u, v), (0, 255, 0), 1)
        txt = f"R {info['angle_right_deg']:+.2f}  F {info['angle_fwd_deg']:+.2f}  " \
              f"tot {info['off_nadir_deg']:.2f} deg"
        cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(out, "no target", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Red circle detector/tracker for a downward Pi camera")
    p.add_argument("--camera", choices=CAMERAS.keys(), default="v2",
                   help="camera module version (sets FOV and default resolution). default v2")
    p.add_argument("--res", type=int, nargs=2, metavar=("W", "H"),
                   help="capture resolution (must be a full-FOV mode or the angles will be wrong)")
    p.add_argument("--hfov", type=float, help="override horizontal FOV in degrees")
    p.add_argument("--vfov", type=float, help="override vertical FOV in degrees")
    p.add_argument("--altitude", type=float, default=None,
                   help="altitude above the target in metres (improves size filtering, gives ground offset)")
    p.add_argument("--image", help="process an image, a folder of images, or a video file instead of the camera")

    g = p.add_argument_group("colour thresholds (OpenCV HSV: H 0-180, S/V 0-255)")
    g.add_argument("--h-low", type=int, default=10, help="red hue upper bound of the low range (0..h_low)")
    g.add_argument("--h-high", type=int, default=160, help="red hue lower bound of the high range (h_high..180)")
    g.add_argument("--s-min", type=int, default=90, help="minimum saturation")
    g.add_argument("--v-min", type=int, default=60, help="minimum brightness")
    g.add_argument("--morph", type=int, default=3, help="morphology kernel size (1 = off)")

    g = p.add_argument_group("shape filtering")
    g.add_argument("--size-tol", type=float, default=2.0,
                   help="allowed diameter ratio vs expected when --altitude is given (2 = 0.5x..2x)")
    g.add_argument("--min-area", type=float, default=12, help="min blob area px (used if no altitude)")
    g.add_argument("--max-area", type=float, default=20000, help="max blob area px (used if no altitude)")
    g.add_argument("--min-fill", type=float, default=0.55, help="min blob fill of its enclosing circle")
    g.add_argument("--max-aspect", type=float, default=1.8, help="max bounding box aspect ratio")

    g = p.add_argument_group("tracking")
    g.add_argument("--track-radius", type=float, default=120, help="ROI half-size / proximity scale in px")
    g.add_argument("--max-misses", type=int, default=10, help="frames without detection before track is dropped")
    g.add_argument("--alpha", type=float, default=0.7, help="position smoothing (1 = raw)")
    g.add_argument("--beta", type=float, default=0.2, help="velocity smoothing")

    g = p.add_argument_group("camera controls")
    g.add_argument("--exposure-us", type=int, help="fixed exposure time (short = less motion blur)")
    g.add_argument("--gain", type=float, default=1.0, help="analogue gain when exposure is fixed")
    g.add_argument("--awb-gains", type=float, nargs=2, metavar=("RED", "BLUE"),
                   help="fixed white balance gains (recommended once tuned: colours stay stable)")
    g.add_argument("--tuning", help="libcamera tuning file (default: the NoIR file for the camera)")
    g.add_argument("--no-noir-tuning", action="store_true", help="use default (non-NoIR) tuning")

    g = p.add_argument_group("output / debugging")
    g.add_argument("--show", action="store_true", help="show a window with the annotated image and mask")
    g.add_argument("--debug-dir", help="save annotated frames + masks here")
    g.add_argument("--debug-every", type=int, default=10, help="save every Nth frame to --debug-dir")
    g.add_argument("--save-raw", type=int, metavar="N", help="just save N raw frames (for HSV tuning) and exit")
    g.add_argument("--quiet-misses", action="store_true", help="do not print lines for frames with no target")
    return p.parse_args()


def main():
    a = parse_args()
    spec = CAMERAS[a.camera]
    hfov = a.hfov or spec["hfov"]
    vfov = a.vfov or spec["vfov"]

    src = FileSource(a.image) if a.image else PiCameraSource(a, a.res or spec["res"], spec["tuning"])

    if a.save_raw:
        os.makedirs("raw_frames", exist_ok=True)
        for i in range(a.save_raw):
            f = src.read()
            if f is None:
                break
            cv2.imwrite(f"raw_frames/frame_{i:04d}.png", f)
            time.sleep(0.5)
        print(f"# saved frames to raw_frames/", file=sys.stderr)
        src.close()
        return

    detector = RedCircleDetector(a)
    cam = tracker = None
    if a.debug_dir:
        os.makedirs(a.debug_dir, exist_ok=True)

    t0 = time.time()
    t_prev = t0
    fps = 0.0
    n = 0
    try:
        while True:
            frame = src.read()
            if frame is None:
                break
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            h, w = frame.shape[:2]
            if cam is None or cam.w != w or cam.h != h:
                cam = CameraModel(w, h, hfov, vfov)
                tracker = Tracker(detector, cam, a)
                if a.altitude:
                    print(f"# {w}x{h}, expected target diameter at {a.altitude} m: "
                          f"{cam.expected_diameter_px(a.altitude):.1f} px", file=sys.stderr)

            exp_d = cam.expected_diameter_px(a.altitude) if a.altitude else None
            det = tracker.update(frame, exp_d)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            rec = {"t": round(now - t0, 3), "found": det is not None,
                   "tracking": tracker.tracking, "fps": round(fps, 1)}
            info = None
            if det:
                # Report the smoothed position; raw detection is in u_raw/v_raw
                u, v = det["u_filt"], det["v_filt"]
                info = cam.angles(u, v, a.altitude, det["diam_px"])
                rec.update(u=round(u, 2), v=round(v, 2),
                           u_raw=round(det["u"], 2), v_raw=round(det["v"], 2),
                           diam_px=round(det["diam_px"], 2), fill=round(det["fill"], 2))
                rec.update({k: round(val, 3) for k, val in info.items()})
            if det or not a.quiet_misses:
                print(json.dumps(rec), flush=True)

            if a.show or (a.debug_dir and n % a.debug_every == 0):
                vis = annotate(frame, cam, det, info)
                if a.debug_dir and n % a.debug_every == 0:
                    cv2.imwrite(os.path.join(a.debug_dir, f"f{n:05d}.jpg"), vis)
                    if tracker.last_mask is not None:
                        cv2.imwrite(os.path.join(a.debug_dir, f"f{n:05d}_mask.png"), tracker.last_mask)
                if a.show:
                    scale = min(1.0, 1280.0 / w)
                    cv2.imshow("tracker", cv2.resize(vis, None, fx=scale, fy=scale))
                    cv2.imshow("mask", cv2.resize(detector.mask(frame), None, fx=scale, fy=scale))
                    key = cv2.waitKey(0 if a.image and not a.image.lower().endswith(
                        (".mp4", ".avi", ".h264", ".mkv")) else 1) & 0xFF
                    if key in (27, ord("q")):
                        break
            n += 1
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if a.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
