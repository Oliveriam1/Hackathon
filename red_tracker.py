#!/usr/bin/env python3
"""
Red-target gimbal tracker for a drone-mounted, downward-looking camera (Raspberry Pi 3)
=====================================================================================
Scenario: 60 x 30 m field, 200 mm red disc lying somewhere on it, drone hovering at ~20 m
near the field centre with RTK/GNSS position. A 2-axis servo gimbal points the camera.

What it does
  SEARCH : steps the gimbal through a grid of aim points that covers the whole field,
           scanning each full-resolution frame for a red blob of the expected size
  TRACK  : follows the target in a small window (ROI) around its last position and
           servos the gimbal so the target sits in the image centre
  LOCKED : target centred -> averages its position on the field (metres) from
           drone position + altitude + heading + gimbal angles

Gimbal geometry assumed ("X/Y" or roll/pitch gimbal, camera looking straight down at 0/0):
  X axis = tilts camera to the drone's RIGHT (+) / left (-)
  Y axis = tilts camera FORWARD (+) / backward (-)
  Mount the camera so the TOP of the image faces the drone's nose.

Install on the Pi:
    sudo apt update
    sudo apt install -y python3-opencv python3-picamera2 python3-numpy pigpio python3-pigpio
    sudo systemctl enable --now pigpiod

Run:
    python3 red_tracker.py                 # full system
    python3 red_tracker.py --show          # + preview window (desktop / VNC)
    python3 red_tracker.py --no-servo      # camera + detection only
    python3 red_tracker.py --tune          # sliders for the red threshold (desktop / VNC)
    python3 red_tracker.py --sim 52 6      # simulation, no hardware: target at field (52 m, 6 m)
"""

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np

# =============================== CONFIG ===============================
# --- camera: OV5647 5MP "night vision" board, 3.6 mm M12 lens ---
# 1296x972 = full field of view, 2x2 binned, fast. (1920x1080 is CROPPED on this sensor - avoid.)
CAPTURE_W, CAPTURE_H = 1296, 972
HFOV_DEG, VFOV_DEG = 54.0, 41.0     # nominal for 3.6 mm on OV5647 - measure yours: HFOV = 2*atan(width_seen / (2*distance))
CAM_ROTATE_180 = False              # set True if the image comes out upside down
# This board has no IR-cut filter -> use the NoIR colour tuning so reds stay red-ish.
# Set to None if you add an IR-cut filter to the lens.
TUNING_FILE = "ov5647_noir.json"

# --- target / detection ---
TARGET_DIAMETER_M = 0.20
# Tuned for grey concrete (R ~= G ~= B), which makes red very easy to separate.
REDNESS_MIN = 25        # pixel is "red" if R - max(G, B) > this ...
RED_FRACTION_MIN = 0.35 # ... AND that difference is > this fraction of R (works in shadow too)
R_MIN = 50              # ... AND R > this (low, so the disc is still found in the drone's shadow)
MIN_AREA_PX = 3         # at 20 m the disc is only a few px, keep this small
SIZE_RATIO = (0.4, 2.5) # accepted blob size vs expected size (rejects red cars, roofs, specks)
MIN_FILL = 0.3          # blob area / bounding-box area (only checked for blobs >= 8 px)

# --- field & drone (field coordinates in metres, origin at one corner) ---
FIELD_W, FIELD_H = 60.0, 30.0
FIELD_MARGIN_M = 2.0
DRONE_X, DRONE_Y = 30.0, 15.0   # used until get_drone_state() is wired to your GNSS
DRONE_ALT_M = 20.0              # height above the field
DRONE_YAW_DEG = 90.0            # nose direction, deg counter-clockwise from field +X (90 = nose along +Y)

# --- gimbal servos (pigpio, BCM pins) ---
X_PIN, Y_PIN = 18, 13
X_CENTER_US, Y_CENTER_US = 1500, 1500   # trim: pulses that make the camera look straight down
US_PER_DEG = 1000.0 / 90.0              # 500..2500 us over 180 deg servo
X_DIR, Y_DIR = 1, 1                     # flip to -1 if an axis moves the wrong way
X_LIMITS = (-60.0, 60.0)                # axis along the 60 m side needs ~58 deg to centre a corner
Y_LIMITS = (-45.0, 45.0)

# --- control ---
KP = 0.45               # fraction of angular error corrected per frame
KD = 0.10
MAX_STEP_DEG = 3.0      # max gimbal move per frame
DEADBAND_DEG = 0.1
LOCK_DEG = 0.4          # |error| below this counts as "centred"
LOCK_FRAMES = 5         # centred this many frames -> LOCKED

# --- search / tracking ---
SEARCH_OVERLAP = 0.7    # grid spacing as a fraction of the camera footprint
SETTLE_S = 0.5          # wait after each search move before grabbing a frame
ROI_HALF_MIN = 120      # tracking window half-size in px
LOST_FRAMES = 15        # frames without target before going back to SEARCH
# --- lightweight contour geometry / event logging ---
ANGLE_LOG_DELTA_DEG = 5.0
EVENT_LOG_MIN_INTERVAL_S = 0.25
QUAD_MIN_AREA_PX = 300
QUAD_MAX_FRAME_FRACTION = 0.95
CIRCLE_MIN_AREA_PX = 12
CIRCLE_MIN_CIRCULARITY = 0.68
CIRCLE_MAX_ASPECT_RATIO = 1.45
# =====================================================================

F_X = (CAPTURE_W / 2) / math.tan(math.radians(HFOV_DEG / 2))  # focal length in px
F_Y = (CAPTURE_H / 2) / math.tan(math.radians(VFOV_DEG / 2))


# --------------------------- drone state ---------------------------
def get_drone_state():
    """Return drone position/attitude in field coordinates.
    Replace with your RTK GNSS / flight-controller readout (lat/lon -> field metres,
    altitude above field, heading, and roll/pitch if available)."""
    return dict(x=DRONE_X, y=DRONE_Y, alt=DRONE_ALT_M, yaw=DRONE_YAW_DEG, roll=0.0, pitch=0.0)


# --------------------------- geometry ---------------------------
def field_to_body(fx, fy, st):
    """Field point -> (right, forward) metres relative to the drone."""
    yaw = math.radians(st["yaw"])
    rx, ry = fx - st["x"], fy - st["y"]
    fwd = rx * math.cos(yaw) + ry * math.sin(yaw)
    right = rx * math.sin(yaw) - ry * math.cos(yaw)
    return right, fwd


def body_to_field(right, fwd, st):
    yaw = math.radians(st["yaw"])
    return (st["x"] + fwd * math.cos(yaw) + right * math.sin(yaw),
            st["y"] + fwd * math.sin(yaw) - right * math.cos(yaw))


def aim_to_ground(ax, ay, h):
    """Gimbal angles (deg) -> ground point (right, fwd) in metres where the camera looks."""
    ax, ay = math.radians(ax), math.radians(ay)
    return h * math.tan(ax), h * math.tan(ay) / math.cos(ax)


def ground_to_aim(right, fwd, h):
    ax = math.atan2(right, h)
    ay = math.atan2(fwd * math.cos(ax), h)
    return math.degrees(ax), math.degrees(ay)


def slant_range(ax, ay, h):
    return h / (math.cos(math.radians(ax)) * math.cos(math.radians(ay)))


def expected_diameter_px(ax, ay, h):
    return F_X * TARGET_DIAMETER_M / slant_range(ax, ay, h)


def pixel_error_to_angles(dx, dy):
    """Pixel offset from centre -> angular offset (deg), right/down positive."""
    return math.degrees(math.atan(dx / F_X)), math.degrees(math.atan(dy / F_Y))


def build_search_grid(st):
    """Aim points (gimbal angles) whose camera footprints cover the field."""
    h = st["alt"]
    step_r = SEARCH_OVERLAP * 2 * h * math.tan(math.radians(HFOV_DEG / 2))
    step_f = SEARCH_OVERLAP * 2 * h * math.tan(math.radians(VFOV_DEG / 2))
    m = FIELD_MARGIN_M
    corners = [field_to_body(x, y, st) for x in (-m, FIELD_W + m) for y in (-m, FIELD_H + m)]
    r_min, r_max = min(c[0] for c in corners), max(c[0] for c in corners)
    f_min, f_max = min(c[1] for c in corners), max(c[1] for c in corners)
    nr = max(1, math.ceil((r_max - r_min) / step_r))
    nf = max(1, math.ceil((f_max - f_min) / step_f))
    grid = []
    for j in range(nf):
        f = f_min + (j + 0.5) * (f_max - f_min) / nf
        row = []
        for i in range(nr):
            r = r_min + (i + 0.5) * (r_max - r_min) / nr
            fx, fy = body_to_field(r, f, st)
            if -m - step_r / 2 <= fx <= FIELD_W + m + step_r / 2 and \
               -m - step_f / 2 <= fy <= FIELD_H + m + step_f / 2:
                ax, ay = ground_to_aim(r, f, h)
                row.append((max(X_LIMITS[0], min(X_LIMITS[1], ax)),
                            max(Y_LIMITS[0], min(Y_LIMITS[1], ay))))
        grid.extend(row if j % 2 == 0 else row[::-1])   # serpentine
    return grid or [(0.0, 0.0)]


# --------------------------- hardware ---------------------------
def strengthen_pink_to_red(frame):
    """Shift pink/magenta pixels toward red; leave the rest of the frame unchanged."""
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    pink = (r >= 120) & (b >= 70) & (r >= g + 20) & (r >= b)
    if np.any(pink):
        r[pink] = 255
        g[pink] = (g[pink].astype(np.uint16) * 45 // 100).astype(np.uint8)
        b[pink] = (b[pink].astype(np.uint16) * 35 // 100).astype(np.uint8)
    return frame


class Camera:
    def __init__(self):
        try:
            from picamera2 import Picamera2
            from libcamera import Transform
            tuning = Picamera2.load_tuning_file(TUNING_FILE) if TUNING_FILE else None
            self.cam = Picamera2(tuning=tuning)
            cfg = self.cam.create_video_configuration(
                main={"size": (CAPTURE_W, CAPTURE_H), "format": "RGB888"},  # BGR order in numpy
                transform=Transform(hflip=CAM_ROTATE_180, vflip=CAM_ROTATE_180),
                buffer_count=3)
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(1.5)  # auto exposure / white balance settle
            self.kind = "picamera2"
        except Exception as e:
            print(f"[camera] Picamera2 unavailable ({e}), using OpenCV device 0")
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            self.kind = "opencv"

    def read(self):
        if self.kind == "picamera2":
            return self.cam.capture_array()
        ok, frame = self.cam.read()
        return frame if ok else None

    def close(self):
        self.cam.stop() if self.kind == "picamera2" else self.cam.release()


class Gimbal:
    """x, y are PHYSICAL angles (deg): x = right, y = forward, 0/0 = straight down."""

    def __init__(self, hardware=True):
        self.x = self.y = 0.0
        self.pi = None
        if hardware:
            import pigpio
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpiod not running: sudo systemctl start pigpiod")
        self._write()

    def move_to(self, x, y):
        self.x = max(X_LIMITS[0], min(X_LIMITS[1], x))
        self.y = max(Y_LIMITS[0], min(Y_LIMITS[1], y))
        self._write()

    def move_by(self, dx, dy):
        self.move_to(self.x + dx, self.y + dy)

    def _write(self):
        if self.pi:
            px = X_CENTER_US + X_DIR * self.x * US_PER_DEG
            py = Y_CENTER_US + Y_DIR * self.y * US_PER_DEG
            self.pi.set_servo_pulsewidth(X_PIN, int(max(500, min(2500, px))))
            self.pi.set_servo_pulsewidth(Y_PIN, int(max(500, min(2500, py))))

    def close(self):
        if self.pi:
            self.pi.set_servo_pulsewidth(X_PIN, 0)
            self.pi.set_servo_pulsewidth(Y_PIN, 0)
            self.pi.stop()


class SimCamera:
    """Renders what the gimbal camera would see: grass, the red disc, and a red car-sized decoy."""

    def __init__(self, gimbal, target_xy, decoy_xy=(15.0, 22.0)):
        self.g, self.target, self.decoy = gimbal, target_xy, decoy_xy
        rng = np.random.default_rng(1)
        # grey concrete: blotchy texture + fine grain, slightly warm (no IR filter)
        blotch = cv2.resize(rng.normal(0, 14, (CAPTURE_H // 40, CAPTURE_W // 40)), (CAPTURE_W, CAPTURE_H))
        grain = rng.normal(0, 7, (CAPTURE_H, CAPTURE_W))
        base = 145 + blotch + grain
        self.bg = np.clip(np.dstack([base - 4, base, base + 8]), 0, 255).astype(np.uint8)
        self.rng = rng

    def _project(self, pts_field, st):
        ax, ay = math.radians(self.g.x), math.radians(self.g.y)
        b = np.array([math.cos(ay) * math.sin(ax), math.sin(ay), -math.cos(ay) * math.cos(ax)])
        r = np.array([math.cos(ax), 0.0, math.sin(ax)])
        u = np.array([-math.sin(ax) * math.sin(ay), math.cos(ay), math.sin(ay) * math.cos(ax)])
        out = []
        for fx, fy in pts_field:
            right, fwd = field_to_body(fx, fy, st)
            p = np.array([right, fwd, -st["alt"]])
            zc = p @ b
            if zc <= 0.1:
                return None
            out.append((CAPTURE_W / 2 + F_X * (p @ r) / zc, CAPTURE_H / 2 - F_Y * (p @ u) / zc))
        return np.array(out)

    def read(self):
        st = get_drone_state()
        img = self.bg.copy()
        dx, dy = self.decoy
        car = self._project([(dx, dy), (dx + 4.5, dy), (dx + 4.5, dy + 1.8), (dx, dy + 1.8)], st)
        if car is not None:
            cv2.fillPoly(img, [np.round(car * 16).astype(np.int32)], (30, 30, 200), cv2.LINE_AA, 4)
        R = TARGET_DIAMETER_M / 2
        circle = [(self.target[0] + R * math.cos(t), self.target[1] + R * math.sin(t))
                  for t in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
        disc = self._project(circle, st)
        if disc is not None:
            cv2.fillPoly(img, [np.round(disc * 16).astype(np.int32)], (40, 40, 210), cv2.LINE_AA, 4)
        # drone shadow (1.5 m) lying across the target
        sh = self._project([(self.target[0] + 0.5 + 1.5 * math.cos(t), self.target[1] + 1.5 * math.sin(t))
                            for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)], st)
        if sh is not None:
            m = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.round(sh).astype(np.int32)], 255)
            img[m > 0] = (img[m > 0] * 0.4).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)  # lens blur
        return img

    def close(self):
        pass


# --------------------------- detection ---------------------------
def normalize_rect_angle(rect):
    """Return tilt in degrees relative to the nearest image axis.

    OpenCV's minAreaRect angle convention differs between versions.  The
    width/height correction below converts it to the orientation of the
    rectangle's long side and the modulo step expresses only the deviation
    from horizontal/vertical.  Result is in [-45, +45): negative = left
    (counter-clockwise on screen), positive = right (clockwise on screen).
    """
    (_, _), (w, h), raw_angle = rect
    if w <= 0.0 or h <= 0.0:
        return None

    angle = float(raw_angle)
    # Typical OpenCV builds return [-90, 0); some return [0, 90).
    # Whichever convention is used, the width/height swap maps the long side
    # to a stable orientation before reducing it to the nearest image axis.
    if w < h:
        angle += 90.0
    angle = (angle + 45.0) % 90.0 - 45.0

    # In image coordinates Y points down, so positive screen rotation is a
    # clockwise/right tilt.  The corrected long-side angle already follows
    # that convention.
    return angle


def _contour_center(contour):
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    (x, y), _ = cv2.minEnclosingCircle(contour)
    return float(x), float(y)


def analyze_object_geometry(frame, prefer=None):
    """Find the main quadrilateral and circular contours inside it.

    This intentionally uses only cheap operations: grayscale conversion,
    Canny edges, contour geometry, polygon tests and circularity.  No temporal
    image filters or expensive model-based processing are used.

    Returns a dict with: contour, angle, center, circles; or None when no
    suitable quadrilateral is visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3, L2gradient=False)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    max_quad_area = frame_area * QUAD_MAX_FRAME_FRACTION
    candidates = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < QUAD_MIN_AREA_PX or area > max_quad_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        contains_preferred = False
        if prefer is not None:
            contains_preferred = cv2.pointPolygonTest(
                approx, (float(prefer[0]), float(prefer[1])), False
            ) >= 0
        # Prefer a quadrilateral containing the currently tracked red target;
        # otherwise fall back to the largest valid quadrilateral.
        candidates.append((1 if contains_preferred else 0, area, approx))

    if not candidates:
        return None

    _, quad_area, quad = max(candidates, key=lambda item: (item[0], item[1]))
    rect = cv2.minAreaRect(quad)
    tilt = normalize_rect_angle(rect)
    quad_center = tuple(map(float, rect[0]))

    circle_candidates = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < CIRCLE_MIN_AREA_PX or area >= quad_area * 0.35:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < CIRCLE_MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / float(min(w, h))
        if aspect > CIRCLE_MAX_ASPECT_RATIO:
            continue

        cx, cy = _contour_center(contour)
        if cv2.pointPolygonTest(quad, (float(cx), float(cy)), False) < 0:
            continue
        circle_candidates.append((area, (cx, cy), contour))

    # An edge image can produce an inner and outer contour for the same ring.
    # Keep only one contour per visible circle by merging close centroids.
    circle_candidates.sort(key=lambda item: item[0], reverse=True)
    circles = []
    for area, center, contour in circle_candidates:
        _, radius = cv2.minEnclosingCircle(contour)
        duplicate = False
        for kept in circles:
            dist = math.hypot(center[0] - kept["center"][0], center[1] - kept["center"][1])
            if dist <= max(4.0, 0.5 * max(radius, kept["radius"])):
                duplicate = True
                break
        if not duplicate:
            circles.append({
                "center": center,
                "radius": float(radius),
                "area": float(area),
                "contour": contour,
            })

    return {
        "contour": quad,
        "rect": rect,
        "angle": tilt,
        "center": quad_center,
        "circles": circles,
    }


def detect(frame, exp_px, roi=None, prefer=None):
    """Find the red blob that best matches the expected size. Returns dict or None.
    roi = (x0, y0, x1, y1) window; prefer = (x, y) favour blobs near this point."""
    x0 = y0 = 0
    img = frame
    if roi is not None:
        x0, y0, x1, y1 = roi
        img = frame[y0:y1, x0:x1]
    b, g, r = cv2.split(img)
    red = cv2.subtract(r, cv2.max(g, b))
    _, m1 = cv2.threshold(red, REDNESS_MIN, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(r, R_MIN, 255, cv2.THRESH_BINARY)
    m3 = cv2.compare(red, cv2.convertScaleAbs(r, alpha=RED_FRACTION_MIN), cv2.CMP_GT)
    mask = cv2.bitwise_and(cv2.bitwise_and(m1, m2), m3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best, best_cost = None, float("inf")
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        major = max(bw, bh)
        ratio = major / exp_px
        if not (SIZE_RATIO[0] <= ratio <= SIZE_RATIO[1]):
            continue
        if major >= 8 and area / float(bw * bh) < MIN_FILL:
            continue
        cost = abs(math.log(ratio))
        if prefer is not None:
            cx, cy = x0 + bx + bw / 2, y0 + by + bh / 2
            cost += math.hypot(cx - prefer[0], cy - prefer[1]) / max(ROI_HALF_MIN, 4 * exp_px)
        if cost < best_cost:
            best_cost, best = cost, (i, bx, by, bw, bh, area)
    if best is None:
        return None, mask

    i, bx, by, bw, bh, area = best
    # redness-weighted centroid -> sub-pixel accuracy on tiny blobs
    patch = red[by:by + bh, bx:bx + bw].astype(np.float32) * (labels[by:by + bh, bx:bx + bw] == i)
    m = cv2.moments(patch)
    cx = x0 + bx + (m["m10"] / m["m00"] if m["m00"] else bw / 2)
    cy = y0 + by + (m["m01"] / m["m00"] if m["m00"] else bh / 2)
    return dict(x=cx, y=cy, w=int(bw), h=int(bh), area=int(area)), mask


# --------------------------- main loop ---------------------------
def tune_mode(cam):
    global REDNESS_MIN, R_MIN, RED_FRACTION_MIN
    win = "tune (q = quit)"
    cv2.namedWindow(win)
    cv2.createTrackbar("REDNESS_MIN", win, REDNESS_MIN, 255, lambda _: None)
    cv2.createTrackbar("R_MIN", win, R_MIN, 255, lambda _: None)
    cv2.createTrackbar("RED_FRACTION_MIN x100", win, int(RED_FRACTION_MIN * 100), 100, lambda _: None)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        frame = strengthen_pink_to_red(frame)
        REDNESS_MIN = cv2.getTrackbarPos("REDNESS_MIN", win)
        R_MIN = cv2.getTrackbarPos("R_MIN", win)
        RED_FRACTION_MIN = cv2.getTrackbarPos("RED_FRACTION_MIN x100", win) / 100.0
        _, mask = detect(frame, 10.0)
        view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(win, cv2.resize(view, None, fx=0.4, fy=0.4))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"REDNESS_MIN = {REDNESS_MIN}\nR_MIN = {R_MIN}\nRED_FRACTION_MIN = {RED_FRACTION_MIN}")
            return


def run(args):
    gimbal = Gimbal(hardware=not (args.no_servo or args.sim))
    cam = SimCamera(gimbal, tuple(args.sim)) if args.sim else Camera()
    if args.tune:
        try:
            tune_mode(cam)
        finally:
            cam.close(); gimbal.close(); cv2.destroyAllWindows()
        return

    st = get_drone_state()
    grid = build_search_grid(st)
    print(f"[search] {len(grid)} aim points cover the field")
    state, gi = "SEARCH", 0
    gimbal.move_to(*grid[0])
    settle_until = time.time() + SETTLE_S
    last_pos, misses, prev_err, centred = None, 0, (0.0, 0.0), 0
    estimates = deque(maxlen=60)
    t_prev, fps = time.time(), 0.0
    t_start = time.time()
    result = None
    last_logged_circle_count = None
    last_logged_angle = None
    last_event_log_time = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = strengthen_pink_to_red(frame)
            now = time.time()
            st = get_drone_state()
            H, W = frame.shape[:2]
            exp_px = expected_diameter_px(gimbal.x, gimbal.y, st["alt"])
            det, line = None, ""

            if state == "SEARCH":
                if now >= settle_until:
                    det, _ = detect(frame, exp_px)
                    if det:
                        state, misses = "TRACK", 0
                        print(f"[search] candidate at aim point {gi} ({gimbal.x:+.1f}, {gimbal.y:+.1f}) deg")
                    else:
                        gi = (gi + 1) % len(grid)
                        gimbal.move_to(*grid[gi])
                        settle_until = now + SETTLE_S
                line = f"SEARCH point {gi + 1}/{len(grid)} aim=({gimbal.x:+5.1f},{gimbal.y:+5.1f})deg"
            else:
                half = int(max(ROI_HALF_MIN, 4 * exp_px))
                px, py = last_pos
                roi = (int(max(0, px - half)), int(max(0, py - half)),
                       int(min(W, px + half)), int(min(H, py + half)))
                det, _ = detect(frame, exp_px, roi=roi, prefer=last_pos)
                if det is None and misses >= 3:          # widen to full frame before giving up
                    det, _ = detect(frame, exp_px, prefer=last_pos)
                if det is None:
                    misses += 1
                    centred = 0
                    if misses > LOST_FRAMES:
                        state, estimates = "SEARCH", deque(maxlen=60)
                        settle_until = now + SETTLE_S
                        print("[track] target lost -> SEARCH")
                    line = f"{state} (target missing {misses})"

            if det is not None:
                last_pos, misses = (det["x"], det["y"]), 0
                dx, dy = det["x"] - W / 2, det["y"] - H / 2
                ex, ey = pixel_error_to_angles(dx, dy)          # right / down, deg
                cos_ay = math.cos(math.radians(gimbal.y))
                err_x, err_y = ex / cos_ay, -ey                 # in gimbal axes

                # where the target is on the field (gimbal angle + residual error, drone attitude corrected)
                ax_t = gimbal.x + err_x - st["roll"]
                ay_t = gimbal.y + err_y + st["pitch"]
                tr, tf = aim_to_ground(ax_t, ay_t, st["alt"])
                tgt_field = body_to_field(tr, tf, st)
                br, bf = aim_to_ground(gimbal.x - st["roll"], gimbal.y + st["pitch"], st["alt"])
                off_m = math.hypot(tr - br, tf - bf)            # ground distance: aim point -> target

                if max(abs(err_x), abs(err_y)) < LOCK_DEG:
                    centred += 1
                    estimates.append(tgt_field)
                else:
                    centred = 0
                state = "LOCKED" if centred >= LOCK_FRAMES else "TRACK"

                # incremental PD control on the gimbal
                sx = KP * err_x + KD * (err_x - prev_err[0]) if abs(err_x) > DEADBAND_DEG else 0.0
                sy = KP * err_y + KD * (err_y - prev_err[1]) if abs(err_y) > DEADBAND_DEG else 0.0
                prev_err = (err_x, err_y)
                gimbal.move_by(max(-MAX_STEP_DEG, min(MAX_STEP_DEG, sx)),
                               max(-MAX_STEP_DEG, min(MAX_STEP_DEG, sy)))

                line = (f"{state:6} dx={dx:+7.1f}px dy={dy:+7.1f}px dist={math.hypot(dx, dy):6.1f}px "
                        f"({off_m:4.2f} m on ground) blob={det['w']}x{det['h']}px exp={exp_px:4.1f}px "
                        f"gimbal=({gimbal.x:+5.1f},{gimbal.y:+5.1f})deg")
                if state == "LOCKED" and estimates:
                    a = np.array(estimates)
                    mx, my = a.mean(axis=0)
                    sd = float(np.hypot(*a.std(axis=0)))
                    result = (mx, my)
                    line += f" | TARGET field=({mx:6.2f}, {my:6.2f}) m +-{sd:.2f}"

            geometry = analyze_object_geometry(frame, prefer=last_pos if det is not None else None)
            if geometry is not None:
                visible_circles = len(geometry["circles"])
                object_angle = geometry["angle"]
                if geometry["circles"]:
                    # Largest circular contour is the most stable centre to log.
                    log_center = geometry["circles"][0]["center"]
                else:
                    log_center = geometry["center"]
            else:
                visible_circles = 0
                object_angle = None
                log_center = last_pos if last_pos is not None else (W / 2.0, H / 2.0)

            count_changed = (last_logged_circle_count is None or
                             visible_circles != last_logged_circle_count)
            angle_changed = (object_angle is not None and
                             (last_logged_angle is None or
                              abs(object_angle - last_logged_angle) > ANGLE_LOG_DELTA_DEG))
            if (count_changed or angle_changed) and now - last_event_log_time >= EVENT_LOG_MIN_INTERVAL_S:
                angle_text = f"{object_angle:+.1f}" if object_angle is not None else "N/A"
                print(f"INFO: Viditelne kruhy: {visible_circles}, "
                      f"Stred: ({int(round(log_center[0]))}, {int(round(log_center[1]))}), "
                      f"Naklon: {angle_text}°")
                last_logged_circle_count = visible_circles
                if object_angle is not None:
                    last_logged_angle = object_angle
                last_event_log_time = now

            dt, t_prev = now - t_prev, now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt
            if args.show:
                view = frame.copy()
                cv2.drawMarker(view, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)
                if det is not None:
                    c = (int(det["x"]), int(det["y"]))
                    cv2.circle(view, c, max(12, int(exp_px)), (0, 255, 0), 2)
                    cv2.line(view, (W // 2, H // 2), c, (0, 255, 255), 1)
                if geometry is not None:
                    cv2.drawContours(view, [geometry["contour"]], -1, (255, 0, 255), 2)
                    for circle in geometry["circles"]:
                        cc = (int(round(circle["center"][0])), int(round(circle["center"][1])))
                        rr = max(3, int(round(circle["radius"])))
                        cv2.circle(view, cc, rr, (0, 255, 0), 2)
                cv2.putText(view, state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(view, f"Viditelne kruhy: {visible_circles}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                tilt_text = f"{object_angle:+.1f}" if object_angle is not None else "N/A"
                cv2.putText(view, f"Naklon objektu: {tilt_text} stupnu", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(view, f"FPS: {fps:.1f}", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("tracker (q = quit)", cv2.resize(view, None, fx=0.5, fy=0.5))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds and now - t_start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        gimbal.close()
        cam.close()
        cv2.destroyAllWindows()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="preview window")
    ap.add_argument("--no-servo", action="store_true", help="don't drive the gimbal")
    ap.add_argument("--tune", action="store_true", help="red threshold sliders")
    ap.add_argument("--sim", nargs=2, type=float, metavar=("X", "Y"),
                    help="simulate (no hardware) with the target at field X, Y metres")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds")
    run(ap.parse_args())
