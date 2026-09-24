#!/usr/bin/env python3
"""
Red-target gimbal tracker for Raspberry Pi 3
--------------------------------------------
- Grabs frames from the ribbon (CSI) camera via Picamera2
- Finds the largest red blob / circle
- Reports its offset from the frame center (pixels, normalized, degrees)
- Drives a 2-axis pan/tilt servo gimbal (PWM via pigpio) to center the target

Install on the Pi (Raspberry Pi OS Bookworm/Bullseye):
    sudo apt update
    sudo apt install -y python3-opencv python3-picamera2 python3-numpy pigpio python3-pigpio
    sudo systemctl enable --now pigpiod

Run:
    python3 red_tracker.py              # track + move gimbal, headless
    python3 red_tracker.py --show       # also show a preview window (needs desktop / VNC)
    python3 red_tracker.py --no-servo   # detection only, gimbal not moved
    python3 red_tracker.py --tune       # sliders to tune the red HSV range (needs desktop)
"""

import argparse
import math
import time

import cv2
import numpy as np

# ============================ CONFIG ============================
FRAME_W, FRAME_H = 640, 480       # capture resolution
PROCESS_SCALE = 0.5               # detect on a downscaled image (faster on Pi 3)

# Camera field of view in degrees - SET THIS FOR YOUR LENS
# (Pi Cam v2: 62.2 x 48.8, Pi Cam v1: 53.5 x 41.4, Pi Cam v3: 66 x 41)
HFOV_DEG = 62.2
VFOV_DEG = 48.8

# Red in HSV wraps around hue 0, so use two ranges
RED_LOWER_1 = np.array([0, 120, 70]);   RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 70]); RED_UPPER_2 = np.array([180, 255, 255])
MIN_AREA_PX = 40                  # ignore blobs smaller than this (in processed image)
MIN_CIRCULARITY = 0.5             # 1.0 = perfect circle; set 0 to accept any shape

# Servos (BCM pin numbers)
PAN_PIN, TILT_PIN = 18, 13
SERVO_MIN_US, SERVO_MAX_US = 500, 2500   # pulse width at -90 deg / +90 deg
PAN_LIMITS = (-80.0, 80.0)               # mechanical limits in degrees
TILT_LIMITS = (-60.0, 45.0)
PAN_DIR, TILT_DIR = 1, 1                 # flip to -1 if an axis moves the wrong way

# Controller (incremental, runs every frame)
KP = 0.35              # fraction of angular error corrected per frame
KD = 0.10              # damping on change of error
MAX_STEP_DEG = 4.0     # max servo move per frame
DEADBAND_DEG = 0.5     # don't move if error is smaller than this
SMOOTHING = 0.5        # EMA on target position (0 = none, 0.9 = heavy)
LOST_HOLD_S = 1.5      # hold position this long after losing target
RETURN_SPEED_DEG = 0.5 # then drift back to center by this much per frame
# ================================================================


class Camera:
    """Picamera2 for the ribbon camera, OpenCV fallback for USB/testing."""

    def __init__(self, w, h):
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(
                main={"size": (w, h), "format": "RGB888"})  # RGB888 arrives as BGR in numpy
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(1.0)  # let auto-exposure settle
            self.kind = "picamera2"
        except Exception as e:
            print(f"[camera] Picamera2 unavailable ({e}), falling back to OpenCV device 0")
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.kind = "opencv"

    def read(self):
        if self.kind == "picamera2":
            return self.cam.capture_array()
        ok, frame = self.cam.read()
        return frame if ok else None

    def close(self):
        if self.kind == "picamera2":
            self.cam.stop()
        else:
            self.cam.release()


class Gimbal:
    """Two hobby servos driven by pigpio (hardware-timed, jitter-free PWM)."""

    def __init__(self, enabled=True):
        self.pan = 0.0
        self.tilt = 0.0
        self.pi = None
        if not enabled:
            return
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("pigpiod not running - run: sudo systemctl start pigpiod")
        self.write()

    @staticmethod
    def _us(angle):
        angle = max(-90.0, min(90.0, angle))
        return int(SERVO_MIN_US + (angle + 90.0) / 180.0 * (SERVO_MAX_US - SERVO_MIN_US))

    def move_by(self, d_pan, d_tilt):
        self.pan = max(PAN_LIMITS[0], min(PAN_LIMITS[1], self.pan + d_pan))
        self.tilt = max(TILT_LIMITS[0], min(TILT_LIMITS[1], self.tilt + d_tilt))
        self.write()

    def write(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(PAN_PIN, self._us(self.pan))
            self.pi.set_servo_pulsewidth(TILT_PIN, self._us(self.tilt))

    def close(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(PAN_PIN, 0)   # stop pulses
            self.pi.set_servo_pulsewidth(TILT_PIN, 0)
            self.pi.stop()


def find_red_target(frame_bgr, lo1=RED_LOWER_1, hi1=RED_UPPER_1, lo2=RED_LOWER_2, hi2=RED_UPPER_2):
    """Return ((x, y), radius, mask) of the best red blob in full-frame pixels, or (None, 0, mask)."""
    small = cv2.resize(frame_bgr, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE,
                       interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0
        if circ < MIN_CIRCULARITY:
            continue
        if area > best_area:
            best, best_area = c, area
    if best is None:
        return None, 0, mask

    m = cv2.moments(best)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    _, r = cv2.minEnclosingCircle(best)
    s = 1.0 / PROCESS_SCALE
    return (cx * s, cy * s), r * s, mask


def pixel_to_angles(dx, dy, w, h):
    """Convert pixel offset from center to angular offset (deg) using the lens FOV."""
    fx = (w / 2) / math.tan(math.radians(HFOV_DEG / 2))
    fy = (h / 2) / math.tan(math.radians(VFOV_DEG / 2))
    return math.degrees(math.atan(dx / fx)), math.degrees(math.atan(dy / fy))


def tune_mode(cam):
    """Interactive HSV sliders. Press q to quit and print values."""
    win = "tune"
    cv2.namedWindow(win)
    for name, val, mx in [("H1lo", 0, 180), ("H1hi", 10, 180), ("H2lo", 170, 180),
                          ("H2hi", 180, 180), ("Smin", 120, 255), ("Vmin", 70, 255)]:
        cv2.createTrackbar(name, win, val, mx, lambda _: None)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        g = lambda n: cv2.getTrackbarPos(n, win)
        lo1 = np.array([g("H1lo"), g("Smin"), g("Vmin")]); hi1 = np.array([g("H1hi"), 255, 255])
        lo2 = np.array([g("H2lo"), g("Smin"), g("Vmin")]); hi2 = np.array([g("H2hi"), 255, 255])
        _, _, mask = find_red_target(frame, lo1, hi1, lo2, hi2)
        cv2.imshow(win, np.hstack([cv2.resize(frame, (mask.shape[1], mask.shape[0])),
                                   cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"RED_LOWER_1 = {lo1.tolist()}  RED_UPPER_1 = {hi1.tolist()}")
            print(f"RED_LOWER_2 = {lo2.tolist()}  RED_UPPER_2 = {hi2.tolist()}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="show preview window")
    ap.add_argument("--no-servo", action="store_true", help="don't drive the gimbal")
    ap.add_argument("--tune", action="store_true", help="HSV tuning sliders")
    args = ap.parse_args()

    cam = Camera(FRAME_W, FRAME_H)
    if args.tune:
        try:
            tune_mode(cam)
        finally:
            cam.close(); cv2.destroyAllWindows()
        return

    gimbal = Gimbal(enabled=not args.no_servo)
    smooth = None
    prev_err = (0.0, 0.0)
    last_seen = 0.0
    t_prev, fps = time.time(), 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            center = (w / 2, h / 2)
            pos, radius, _ = find_red_target(frame)
            now = time.time()

            if pos is not None:
                last_seen = now
                smooth = pos if smooth is None else (
                    SMOOTHING * smooth[0] + (1 - SMOOTHING) * pos[0],
                    SMOOTHING * smooth[1] + (1 - SMOOTHING) * pos[1])
                dx = smooth[0] - center[0]          # + = target right of center
                dy = smooth[1] - center[1]          # + = target below center
                dist_px = math.hypot(dx, dy)
                ax, ay = pixel_to_angles(dx, dy, w, h)

                # Incremental PD: nudge the gimbal toward the target each frame
                d_pan = d_tilt = 0.0
                if abs(ax) > DEADBAND_DEG:
                    d_pan = KP * ax + KD * (ax - prev_err[0])
                if abs(ay) > DEADBAND_DEG:
                    d_tilt = KP * ay + KD * (ay - prev_err[1])
                d_pan = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, d_pan))
                d_tilt = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, d_tilt))
                gimbal.move_by(PAN_DIR * d_pan, -TILT_DIR * d_tilt)  # image y grows downward
                prev_err = (ax, ay)

                print(f"TARGET dx={dx:+6.1f}px dy={dy:+6.1f}px dist={dist_px:6.1f}px "
                      f"| err={ax:+5.1f}deg,{ay:+5.1f}deg "
                      f"| gimbal pan={gimbal.pan:+5.1f} tilt={gimbal.tilt:+5.1f} | {fps:4.1f}fps")
            else:
                smooth = None
                prev_err = (0.0, 0.0)
                if now - last_seen > LOST_HOLD_S:
                    # slowly return to center
                    gimbal.move_by(-math.copysign(min(RETURN_SPEED_DEG, abs(gimbal.pan)), gimbal.pan),
                                   -math.copysign(min(RETURN_SPEED_DEG, abs(gimbal.tilt)), gimbal.tilt))
                print(f"NO TARGET | gimbal pan={gimbal.pan:+5.1f} tilt={gimbal.tilt:+5.1f} | {fps:4.1f}fps")

            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1 / dt)

            if args.show:
                cx, cy = int(center[0]), int(center[1])
                cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
                if pos is not None:
                    p = (int(smooth[0]), int(smooth[1]))
                    cv2.circle(frame, p, int(radius), (0, 255, 0), 2)
                    cv2.line(frame, (cx, cy), p, (0, 255, 255), 1)
                cv2.imshow("tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        gimbal.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
