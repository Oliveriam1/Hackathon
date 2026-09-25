#!/usr/bin/env python3
"""
Red-target gimbal tracker for a drone-mounted, downward-looking camera (Raspberry Pi 3)
=====================================================================================
Scenario: 60 x 30 m concrete field, 200 mm red disc lying somewhere on it, drone hovering
at ~20 m with RTK/GNSS position. A 2-axis servo gimbal points the camera.

  SEARCH : steps the gimbal through aim points covering the field and looks for the disc
  TRACK  : follows the disc in a window around its last position, servos it to the centre
  LOCKED : disc centred -> averages its field position (metres)

Detection (v3)
  * Colour first: one rule covers both real red and the pink/magenta that the NoIR camera
    makes of red ink:  R clearly above G,  B not far below G (rejects orange/brown/wood).
  * Every coloured blob is checked as a FILLED ELLIPSE (a tilted circle is an ellipse):
    aspect up to 4.5:1 (~77 deg tilt), silhouette must match its fitted ellipse,
    glare holes inside the disc are filled in.
  * Round shapes inside other shapes (circle printed on white paper) are found; the paper
    outline no longer hides the circle.
  * While tracking, the size filter follows the disc's own last size, so it keeps working
    when the disc gets closer/further or tilts. The altitude-based size is used only to
    find it the first time.
  * If colour briefly fails (glare, shadow) a shape-only check near the last position
    keeps the track alive.

Control (v3)
  * Latency-compensated: each frame's error is added to the gimbal angle AT THE TIME THE
    FRAME WAS TAKEN (not the current one). This removes the overshoot/oscillation caused by
    camera + servo delay, so the gimbal settles and LOCK is reached.
  * Lock with hysteresis and a short coast period, so one bad frame doesn't drop the lock.

Gimbal geometry ("X/Y" roll/pitch gimbal, camera straight down at 0/0):
  X axis = tilts camera to the drone's RIGHT (+),  Y axis = tilts camera FORWARD (+).
  Top of the image faces the drone's nose.

Install on the Pi:
    sudo apt install -y python3-opencv python3-picamera2 python3-numpy pigpio python3-pigpio
    sudo systemctl enable --now pigpiod

Run:
    python3 red_tracker.py                      # full system (drone)
    python3 red_tracker.py --bench 1.5 --show   # desk test: printed disc ~1.5 m from camera
    python3 red_tracker.py --bench 1.5 --show --no-servo   # desk test, gimbal not moved
    python3 red_tracker.py --tune               # colour sliders (desktop / VNC)
    python3 red_tracker.py --sim 52 6           # simulation, no hardware
    python3 red_tracker.py --sim 30.3 15.2 --bench 1.5 --tilt 60 --show   # simulated desk test
"""

import argparse
import math
import time
from collections import deque

import cv2
import numpy as np

# =============================== CONFIG ===============================
# --- camera: OV5647 5MP "night vision" board, 3.6 mm M12 lens ---
CAPTURE_W, CAPTURE_H = 1296, 972    # full FOV, 2x2 binned (1920x1080 is cropped - avoid)
HFOV_DEG, VFOV_DEG = 54.0, 41.0     # measure yours: HFOV = 2*atan(width_seen / (2*distance))
CAM_ROTATE_180 = False
TUNING_FILE = "ov5647_noir.json"    # None if an IR-cut filter is fitted
FRAME_LATENCY_S = 0.10              # capture -> frame in Python (approx. 1-2 frames on a Pi 3)

# --- target colour: red AND NoIR-pink in one rule ---
TARGET_DIAMETER_M = 0.20
R_MIN = 45              # R > this (low -> still works in shadow)
RG_MIN = 28             # R - G > this ...
RG_FRAC = 0.22          # ... and R - G > this fraction of R (brightness independent)
BG_TOL = 22             # B >= G - this   (red: B~G, pink: B>G; orange/brown/wood: B<<G -> rejected)
RB_TOL = 40             # R >= B - this   (rejects blue/purple)

# --- target shape ---
MIN_AREA_PX = 3
SMALL_BLOB_PX = 10          # below this size only a basic aspect check is possible
MAX_ASPECT = 4.5            # ellipse major/minor; 4.5 = disc tilted ~77 deg
MIN_ELLIPSE_IOU = 0.72      # filled silhouette vs its fitted ellipse
MIN_SOLIDITY = 0.85
MIN_COLOUR_COVERAGE = 0.45  # coloured pixels / silhouette (glare holes allowed)
SEARCH_SIZE_RATIO = (0.3, 3.0)   # vs altitude-based expected size (first detection)
TRACK_SIZE_RATIO = (0.5, 2.0)    # vs the disc's own size in the previous frames

# --- field & drone (field coordinates in metres) ---
FIELD_W, FIELD_H = 60.0, 30.0
FIELD_MARGIN_M = 2.0
DRONE_X, DRONE_Y = 30.0, 15.0
DRONE_ALT_M = 20.0
DRONE_YAW_DEG = 90.0

# --- gimbal servos (pigpio, BCM pins) ---
X_PIN, Y_PIN = 20, 21
X_CENTER_US, Y_CENTER_US = 1500, 1500
US_PER_DEG = 1000.0 / 90.0
X_DIR, Y_DIR = 1, 1
X_LIMITS = (-60.0, 60.0)
Y_LIMITS = (-45.0, 45.0)
SERVO_LAG_S = 0.15          # time for a small servo move to complete

# --- control ---
K_TRACK = 0.7               # fraction of the (latency-compensated) error corrected per frame
TARGET_SMOOTH = 0.3         # EMA on the target direction estimate (0 = off)
MAX_STEP_DEG = 4.0
DEADBAND_DEG = 0.05
LOCK_DEG = 0.4              # centred if error < max(LOCK_DEG, LOCK_RADIUS_FRAC * disc radius)
LOCK_RADIUS_FRAC = 0.35
LOCK_FRAMES = 4             # frames centred to enter LOCKED
UNLOCK_FACTOR = 2.5         # leave LOCKED only if error > this x lock threshold ...
UNLOCK_FRAMES = 3           # ... for this many frames
COAST_FRAMES = 5            # missed frames tolerated without dropping TRACK/LOCKED

# --- search / tracking ---
SEARCH_OVERLAP = 0.7
SETTLE_S = 0.5
ROI_HALF_MIN = 120
LOST_FRAMES = 20
PRINT_EVERY_S = 0.2

# --- contour geometry / event logging (quad + circles, "Naklon") ---
ANGLE_LOG_DELTA_DEG = 5.0
EVENT_LOG_MIN_INTERVAL_S = 0.25
QUAD_MIN_AREA_PX = 300
QUAD_MAX_FRAME_FRACTION = 0.95
CIRCLE_MIN_AREA_PX = 12
CIRCLE_MIN_CIRCULARITY = 0.45     # was 0.68: tilted circles (ellipses) score lower
CIRCLE_MAX_ASPECT_RATIO = 4.5     # was 1.45 on the axis-aligned box: rejected any tilt

# --- performance ---
DETECT_FULL_SCALE = 0.67    # full-frame search is downscaled; tracking ROI is full-res
GEOMETRY_SCALE = 0.50
GEOMETRY_EVERY_N_FRAMES = 3
DISPLAY_SCALE = 0.50
MORPH_KERNEL_3 = np.ones((3, 3), np.uint8)
# =====================================================================

F_X = (CAPTURE_W / 2) / math.tan(math.radians(HFOV_DEG / 2))
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
                buffer_count=2)
            self.cam.configure(cfg)
            self.cam.start()
            time.sleep(1.5)
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
    """x, y = commanded PHYSICAL angles (deg): x = right, y = forward, 0/0 = straight down.
    Keeps a short command history so the controller can ask where the gimbal actually
    pointed when a given frame was captured (servo + camera latency compensation).
    frozen=True (--no-servo): angles never change, so the measured error stays honest."""

    def __init__(self, hardware=True, frozen=False):
        self.x = self.y = 0.0
        self.frozen = frozen
        self.pi = None
        self.hist = deque(maxlen=200)
        self.hist.append((0.0, 0.0, 0.0))
        if hardware and not frozen:
            import pigpio
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpiod not running: sudo systemctl start pigpiod")
        self._write()

    def move_to(self, x, y):
        if self.frozen:
            return
        self.x = max(X_LIMITS[0], min(X_LIMITS[1], x))
        self.y = max(Y_LIMITS[0], min(Y_LIMITS[1], y))
        self.hist.append((time.time(), self.x, self.y))
        self._write()

    def move_by(self, dx, dy):
        self.move_to(self.x + dx, self.y + dy)

    def angle_at(self, t, lag=None):
        """Physical angle at time t: the command that was active SERVO_LAG_S earlier."""
        t -= SERVO_LAG_S if lag is None else lag
        ax, ay = self.hist[0][1], self.hist[0][2]
        for ts, hx, hy in self.hist:
            if ts > t:
                break
            ax, ay = hx, hy
        return ax, ay

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
    """Renders the gimbal camera view: concrete, a red disc printed on white paper (optionally
    tilted, NoIR-pink, with glare), a red car-sized decoy, a drone shadow, and realistic
    servo + camera delay and frame rate."""

    SIM_FPS = 12.0
    SIM_TRUE_DELAY_S = 0.20     # real (unknown) servo + camera delay; deliberately != config

    def __init__(self, gimbal, target_xy, tilt=0.0, pink=False, glare=False,
                 decoy_xy=(15.0, 22.0)):
        self.g, self.target, self.decoy = gimbal, target_xy, decoy_xy
        self.tilt, self.pink, self.glare = tilt, pink, glare
        rng = np.random.default_rng(1)
        blotch = cv2.resize(rng.normal(0, 14, (CAPTURE_H // 40, CAPTURE_W // 40)), (CAPTURE_W, CAPTURE_H))
        base = 145 + blotch + rng.normal(0, 7, (CAPTURE_H, CAPTURE_W))
        self.bg = np.clip(np.dstack([base - 4, base, base + 8]), 0, 255).astype(np.uint8)
        self.t_last = 0.0

    def _project(self, pts_field, st, ang):
        ax, ay = math.radians(ang[0]), math.radians(ang[1])
        b = np.array([math.cos(ay) * math.sin(ax), math.sin(ay), -math.cos(ay) * math.cos(ax)])
        r = np.array([math.cos(ax), 0.0, math.sin(ax)])
        u = np.array([-math.sin(ax) * math.sin(ay), math.cos(ay), math.sin(ay) * math.cos(ax)])
        out = []
        for fx, fy in pts_field:
            right, fwd = field_to_body(fx, fy, st)
            p = np.array([right, fwd, -st["alt"]])
            zc = p @ b
            if zc <= 0.05:
                return None
            out.append((CAPTURE_W / 2 + F_X * (p @ r) / zc, CAPTURE_H / 2 - F_Y * (p @ u) / zc))
        return np.array(out)

    def _fill(self, img, pts, colour):
        if pts is not None and np.all(np.abs(pts) < 1e5):
            cv2.fillPoly(img, [np.round(pts * 16).astype(np.int32)], colour, cv2.LINE_AA, 4)

    def read(self):
        wait = 1.0 / self.SIM_FPS - (time.time() - self.t_last)
        if wait > 0:
            time.sleep(wait)
        self.t_last = time.time()
        st = get_drone_state()
        ang = self.g.angle_at(time.time(), lag=self.SIM_TRUE_DELAY_S)
        img = self.bg.copy()
        dx, dy = self.decoy
        self._fill(img, self._project([(dx, dy), (dx + 4.5, dy), (dx + 4.5, dy + 1.8), (dx, dy + 1.8)], st, ang),
                   (30, 30, 200))
        tx, ty = self.target
        k = math.cos(math.radians(self.tilt))          # paper tilted about the field X axis
        s = 0.15
        self._fill(img, self._project([(tx - s, ty - s * k), (tx + s, ty - s * k), (tx + s, ty + s * k),
                                       (tx - s, ty + s * k)], st, ang), (235, 238, 240))
        R = TARGET_DIAMETER_M / 2
        t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        disc = self._project(list(zip(tx + R * np.cos(t), ty + R * k * np.sin(t))), st, ang)
        self._fill(img, disc, (150, 115, 215) if self.pink else (40, 40, 210))
        if self.glare:
            gl = self._project([(tx + 0.03 + 0.025 * math.cos(a), ty + 0.02 * k + 0.015 * k * math.sin(a))
                                for a in t], st, ang)
            self._fill(img, gl, (250, 250, 250))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def close(self):
        pass


# --------------------------- detection ---------------------------
def colour_mask(img):
    """Binary mask of red OR NoIR-pink pixels (see CONFIG for the rule)."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = cv2.subtract(r, g)
    m = cv2.compare(rg, RG_MIN, cv2.CMP_GT)
    m &= cv2.compare(rg, cv2.convertScaleAbs(r, alpha=RG_FRAC), cv2.CMP_GT)
    m &= cv2.compare(r, R_MIN, cv2.CMP_GT)
    m &= cv2.compare(cv2.add(b, BG_TOL), g, cv2.CMP_GE)
    m &= cv2.compare(cv2.add(r, RB_TOL), b, cv2.CMP_GE)
    return m


def _ellipse_check(contour, bw, bh, ox, oy):
    """Shape test on a closed contour (coordinates relative to ox, oy).
    Returns dict(major, minor, cx, cy, iou, angle, sil) or None if it isn't ellipse-like."""
    area = abs(cv2.contourArea(contour))
    if max(bw, bh) < SMALL_BLOB_PX or len(contour) < 5:
        # tiny blob: only size/aspect are meaningful
        major, minor = float(max(bw, bh)), float(max(1, min(bw, bh)))
        if major / minor > MAX_ASPECT:
            return None
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        else:
            cx, cy = bw / 2.0, bh / 2.0
        return dict(major=major, minor=minor, cx=cx + ox, cy=cy + oy, iou=0.85, angle=0.0, sil=None)

    (ex, ey), (ew, eh), ang = cv2.fitEllipse(contour)
    major, minor = max(ew, eh), min(ew, eh)
    major_angle = ang + 90.0 if ew < eh else ang     # orientation of the major axis
    if minor <= 0 or major / minor > MAX_ASPECT:
        return None
    hull_area = abs(cv2.contourArea(cv2.convexHull(contour)))
    if hull_area <= 0 or area / hull_area < MIN_SOLIDITY:
        return None
    sil = np.zeros((bh, bw), np.uint8)
    cv2.drawContours(sil, [contour], -1, 255, cv2.FILLED)          # fills glare holes
    ell = np.zeros((bh, bw), np.uint8)
    cv2.ellipse(ell, ((ex, ey), (ew, eh), ang), 255, cv2.FILLED)
    union = cv2.countNonZero(sil | ell)
    iou = cv2.countNonZero(sil & ell) / float(union) if union else 0.0
    if iou < MIN_ELLIPSE_IOU:
        return None
    return dict(major=major, minor=minor, cx=ex + ox, cy=ey + oy, iou=iou, angle=major_angle, sil=sil)


def _score(c, size_ref, size_range, prefer, prefer_scale):
    ratio = c["major"] / max(size_ref, 1e-6)
    if not (size_range[0] <= ratio <= size_range[1]):
        return None
    cost = abs(math.log(ratio)) + 2.0 * (1.0 - c["iou"])
    if prefer is not None:
        cost += math.hypot(c["cx"] - prefer[0], c["cy"] - prefer[1]) / max(prefer_scale, 1.0)
    return cost



def _shape_candidates(img, ref, size_range, prefer=None, prefer_scale=1.0):
    """Yield ellipse-like contours independent of colour.

    Used as a fallback in SEARCH/bench and TRACK. The expected 200 mm target
    size still constrains the candidate, so arbitrary circles in the scene are
    not accepted without limit.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, MORPH_KERNEL_3)

    cnts, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    out = []
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        major_bb = max(bw, bh)
        minor_bb = max(1, min(bw, bh))

        if major_bb < max(SMALL_BLOB_PX, size_range[0] * ref * 0.65):
            continue
        if major_bb > size_range[1] * ref * 1.35:
            continue
        if major_bb / float(minor_bb) > MAX_ASPECT:
            continue

        per = cv2.arcLength(cnt, True)
        if per <= 0:
            continue

        # Reject obvious rectangles/squares and jagged reflections.
        approx = cv2.approxPolyDP(cnt, 0.02 * per, True)
        if len(approx) < 6:
            continue

        local = cnt - np.array([bx, by])
        c = _ellipse_check(local, bw, bh, bx, by)
        if c is None:
            continue

        # Shape-only candidates should match the fitted ellipse very well.
        if c["iou"] < max(MIN_ELLIPSE_IOU, 0.80):
            continue

        cost = _score(c, ref, size_range, prefer, prefer_scale)
        if cost is None:
            continue

        c["source"] = "shape"
        out.append((cost, c))

    return out


def detect(frame, size_ref, roi=None, prefer=None, size_range=SEARCH_SIZE_RATIO, shape_fallback=False):
    """Find the red/pink disc (circle or tilted ellipse).

    size_ref   : expected major-axis size in px (altitude-based in SEARCH, last size in TRACK)
    roi        : (x0, y0, x1, y1) search window, full resolution
    prefer     : (x, y) favour candidates near this point
    Returns (det or None, colour_mask). det has x, y, w (major), h (minor), tilt, angle, source.
    """
    x0 = y0 = 0
    img = frame
    if roi is not None:
        x0, y0, x1, y1 = roi
        img = frame[y0:y1, x0:x1]
    scale = 1.0
    if roi is None and DETECT_FULL_SCALE < 0.999:
        scale = DETECT_FULL_SCALE
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ref = size_ref * scale
    pref = None if prefer is None else ((prefer[0] - x0) * scale, (prefer[1] - y0) * scale)
    pscale = max(ROI_HALF_MIN * scale, 3.0 * ref)

    mask = colour_mask(img)
    k = 5 if ref > 40 else 3                       # bridge small glare gaps on big discs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best, best_cost = None, float("inf")
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        major_bb = max(bw, bh)
        if major_bb < size_range[0] * ref * 0.7 or min(bw, bh) > size_range[1] * ref * 1.3:
            continue                               # cheap size pre-filter
        comp = (labels[by:by + bh, bx:bx + bw] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        c = _ellipse_check(cnt, bw, bh, bx, by)
        if c is None:
            continue
        if c["sil"] is not None:
            cover = cv2.countNonZero(comp & c["sil"]) / float(max(1, cv2.countNonZero(c["sil"])))
            if cover < MIN_COLOUR_COVERAGE:
                continue
        cost = _score(c, ref, size_range, pref, pscale)
        if cost is not None and cost < best_cost:
            best_cost, best, c["source"] = cost, c, "colour"

    # Shape fallback:
    # - in TRACK, keep the target alive when colour is lost by glare/shadow;
    # - in SEARCH/bench, allow a clearly round 200 mm surface to be acquired
    #   even when the NoIR colour rendering is poor.
    use_shape = shape_fallback or (roi is None)
    if best is None and use_shape:
        shape_range = (0.45, 1.8) if (shape_fallback and pref is not None) else size_range
        for cost, c in _shape_candidates(
                img, ref, shape_range, prefer=pref, prefer_scale=pscale):
            if pref is not None:
                if math.hypot(c["cx"] - pref[0], c["cy"] - pref[1]) > 1.25 * max(ref, 1.0):
                    continue
            if cost < best_cost:
                best_cost, best = cost, c

    if best is None:
        return None, mask
    inv = 1.0 / scale
    return dict(x=x0 + best["cx"] * inv, y=y0 + best["cy"] * inv,
                w=best["major"] * inv, h=best["minor"] * inv,
                tilt=math.degrees(math.acos(min(1.0, best["minor"] / max(best["major"], 1e-6)))),
                angle=best["angle"], source=best["source"]), mask


def normalize_rect_angle(rect):
    """Tilt in degrees relative to the nearest image axis, in [-45, +45)."""
    (_, _), (w, h), raw_angle = rect
    if w <= 0.0 or h <= 0.0:
        return None
    angle = float(raw_angle)
    if w < h:
        angle += 90.0
    return (angle + 45.0) % 90.0 - 45.0


def _contour_center(contour):
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    (x, y), _ = cv2.minEnclosingCircle(contour)
    return float(x), float(y)


def analyze_object_geometry(frame, prefer=None):
    """Find the main quadrilateral and circular contours inside it.

    The expensive edge/contour work is performed on a downscaled frame and all
    coordinates are mapped back to the original frame before returning.
    """
    scale = float(GEOMETRY_SCALE)
    if not (0.0 < scale <= 1.0):
        scale = 1.0

    if scale < 0.999:
        work = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        prefer_work = ((prefer[0] * scale, prefer[1] * scale)
                       if prefer is not None else None)
    else:
        work = frame
        prefer_work = prefer

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3, L2gradient=False)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    area_scale = scale * scale
    frame_area = work.shape[0] * work.shape[1]
    min_quad_area = max(4.0, QUAD_MIN_AREA_PX * area_scale)
    max_quad_area = frame_area * QUAD_MAX_FRAME_FRACTION
    min_circle_area = max(2.0, CIRCLE_MIN_AREA_PX * area_scale)
    candidates = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < min_quad_area or area > max_quad_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        contains_preferred = False
        if prefer_work is not None:
            contains_preferred = cv2.pointPolygonTest(
                approx, (float(prefer_work[0]), float(prefer_work[1])), False
            ) >= 0
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
        if area < min_circle_area or area >= quad_area * 0.35:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < CIRCLE_MIN_CIRCULARITY:
            continue

        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)   # rotated box: tilt-safe
        if rw <= 0 or rh <= 0:
            continue
        aspect = max(rw, rh) / float(min(rw, rh))
        if aspect > CIRCLE_MAX_ASPECT_RATIO:
            continue

        cx, cy = _contour_center(contour)
        if cv2.pointPolygonTest(quad, (float(cx), float(cy)), False) < 0:
            continue
        circle_candidates.append((area, (cx, cy), contour))

    circle_candidates.sort(key=lambda item: item[0], reverse=True)
    circles = []
    for area, center, contour in circle_candidates:
        _, radius = cv2.minEnclosingCircle(contour)
        duplicate = False
        for kept in circles:
            dist = math.hypot(center[0] - kept["center_work"][0],
                              center[1] - kept["center_work"][1])
            if dist <= max(4.0 * scale, 0.5 * max(radius, kept["radius_work"])):
                duplicate = True
                break
        if not duplicate:
            circles.append({
                "center_work": center,
                "radius_work": float(radius),
                "area_work": float(area),
                "contour_work": contour,
            })

    inv = 1.0 / scale
    quad_out = np.rint(quad.astype(np.float32) * inv).astype(np.int32)
    rect_out = (
        (rect[0][0] * inv, rect[0][1] * inv),
        (rect[1][0] * inv, rect[1][1] * inv),
        rect[2],
    )
    circles_out = []
    for circle in circles:
        circles_out.append({
            "center": (circle["center_work"][0] * inv,
                       circle["center_work"][1] * inv),
            "radius": circle["radius_work"] * inv,
            "area": circle["area_work"] * inv * inv,
            "contour": np.rint(
                circle["contour_work"].astype(np.float32) * inv
            ).astype(np.int32),
        })

    return {
        "contour": quad_out,
        "rect": rect_out,
        "angle": tilt,
        "center": (quad_center[0] * inv, quad_center[1] * inv),
        "circles": circles_out,
    }



# --------------------------- main loop ---------------------------
def tune_mode(cam):
    global R_MIN, RG_MIN, RG_FRAC, BG_TOL
    win = "tune (q = quit)"
    cv2.namedWindow(win)
    cv2.createTrackbar("R_MIN", win, R_MIN, 255, lambda _: None)
    cv2.createTrackbar("RG_MIN", win, RG_MIN, 255, lambda _: None)
    cv2.createTrackbar("RG_FRAC x100", win, int(RG_FRAC * 100), 100, lambda _: None)
    cv2.createTrackbar("BG_TOL", win, BG_TOL, 100, lambda _: None)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        R_MIN = cv2.getTrackbarPos("R_MIN", win)
        RG_MIN = cv2.getTrackbarPos("RG_MIN", win)
        RG_FRAC = cv2.getTrackbarPos("RG_FRAC x100", win) / 100.0
        BG_TOL = cv2.getTrackbarPos("BG_TOL", win)
        mask = colour_mask(frame)
        view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(win, cv2.resize(view, None, fx=0.4, fy=0.4))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"R_MIN = {R_MIN}\nRG_MIN = {RG_MIN}\nRG_FRAC = {RG_FRAC}\nBG_TOL = {BG_TOL}")
            return


def _close_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass                                  # OpenCV without GUI support (headless)


def run(args):
    global DRONE_ALT_M
    if args.bench:
        DRONE_ALT_M = args.bench            # camera-to-target distance on the desk
    frozen = args.no_servo
    gimbal = Gimbal(hardware=not args.sim, frozen=frozen)
    if args.sim:
        cam = SimCamera(gimbal, tuple(args.sim), tilt=args.tilt, pink=args.pink, glare=args.glare)
    else:
        cam = Camera()
    if args.tune:
        try:
            tune_mode(cam)
        finally:
            cam.close(); gimbal.close(); _close_windows()
        return None

    st = get_drone_state()
    grid = [(0.0, 0.0)] if (args.bench or frozen) else build_search_grid(st)
    print(f"[search] {len(grid)} aim point(s)" + ("  (bench / no-servo: camera stays centred)" if len(grid) == 1 else ""))
    state, gi = "SEARCH", 0
    gimbal.move_to(*grid[0])
    settle_until = time.time() + SETTLE_S
    last_pos, last_size, misses = None, None, 0
    tgt_abs = None                          # smoothed absolute target direction (gimbal angles)
    centred, off_count = 0, 0
    estimates = deque(maxlen=60)
    t_prev, fps, t_print, t_start = time.time(), 0.0, 0.0, time.time()
    t_first_lock = None
    result = None
    last_logged_circle_count, last_logged_angle, last_event_log_time = None, None, 0.0
    geometry, geometry_frame_counter = None, 0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            t_cap = now - FRAME_LATENCY_S
            st = get_drone_state()
            H, W = frame.shape[:2]
            gx, gy = (gimbal.x, gimbal.y) if frozen else gimbal.angle_at(t_cap)
            exp_px = expected_diameter_px(gx, gy, st["alt"])
            det, line = None, ""

            if state == "SEARCH":
                if now >= settle_until:
                    det, _ = detect(frame, exp_px, size_range=SEARCH_SIZE_RATIO)
                    if det:
                        state, misses, centred, off_count = "TRACK", 0, 0, 0
                        last_size, tgt_abs = det["w"], None
                        print(f"[search] target found at aim point {gi} ({gimbal.x:+.1f}, {gimbal.y:+.1f}) deg")
                    elif len(grid) > 1:
                        gi = (gi + 1) % len(grid)
                        gimbal.move_to(*grid[gi])
                        settle_until = now + SETTLE_S
                line = f"SEARCH point {gi + 1}/{len(grid)} aim=({gimbal.x:+5.1f},{gimbal.y:+5.1f})deg"
            else:
                half = int(max(ROI_HALF_MIN, 2.5 * last_size, 3 * exp_px))
                px, py = last_pos
                roi = (int(max(0, px - half)), int(max(0, py - half)),
                       int(min(W, px + half)), int(min(H, py + half)))
                det, _ = detect(frame, last_size, roi=roi, prefer=last_pos,
                                size_range=TRACK_SIZE_RATIO, shape_fallback=True)
                if det is None and misses >= 2:
                    det, _ = detect(frame, last_size, prefer=last_pos, size_range=TRACK_SIZE_RATIO)
                if det is None:
                    misses += 1
                    if misses > COAST_FRAMES:
                        centred = 0
                        if state == "LOCKED":
                            state = "TRACK"
                    if misses > LOST_FRAMES:
                        state, estimates, tgt_abs = "SEARCH", deque(maxlen=60), None
                        settle_until = now + SETTLE_S
                        print("[track] target lost -> SEARCH")
                    line = f"{state} (target missing {misses})"

            if det is not None:
                if state == "SEARCH":
                    state = "TRACK"
                last_pos, misses = (det["x"], det["y"]), 0
                last_size = det["w"] if last_size is None else 0.6 * last_size + 0.4 * det["w"]
                dx, dy = det["x"] - W / 2, det["y"] - H / 2
                ex, ey = pixel_error_to_angles(dx, dy)
                err_x = ex / math.cos(math.radians(gy))
                err_y = -ey

                # absolute direction of the target = where the gimbal pointed when the frame
                # was taken + the error seen in that frame  (latency compensation)
                ax_abs, ay_abs = gx + err_x, gy + err_y
                if tgt_abs is None:
                    tgt_abs = (ax_abs, ay_abs)
                else:
                    a = TARGET_SMOOTH
                    tgt_abs = (a * tgt_abs[0] + (1 - a) * ax_abs, a * tgt_abs[1] + (1 - a) * ay_abs)

                # ground position of the target
                tr, tf = aim_to_ground(ax_abs - st["roll"], ay_abs + st["pitch"], st["alt"])
                tgt_field = body_to_field(tr, tf, st)
                br, bf = aim_to_ground(gx - st["roll"], gy + st["pitch"], st["alt"])
                off_m = math.hypot(tr - br, tf - bf)

                # lock logic with hysteresis
                disc_radius_deg = math.degrees(math.atan(det["w"] / 2 / F_X))
                lock_thr = max(LOCK_DEG, LOCK_RADIUS_FRAC * disc_radius_deg)
                err = max(abs(err_x), abs(err_y))
                if frozen or err < lock_thr:           # no gimbal: "locked" = stable detection
                    centred, off_count = centred + 1, 0
                else:
                    off_count += 1
                    if state != "LOCKED":
                        centred = 0
                    elif err > UNLOCK_FACTOR * lock_thr or off_count >= UNLOCK_FRAMES:
                        centred, state = 0, "TRACK"
                if state != "LOCKED" and centred >= LOCK_FRAMES:
                    state = "LOCKED"
                    t_first_lock = t_first_lock or now
                if state == "LOCKED":
                    estimates.append(tgt_field)

                # move gimbal toward the absolute target direction
                if not frozen:
                    sx = K_TRACK * (tgt_abs[0] - gimbal.x)
                    sy = K_TRACK * (tgt_abs[1] - gimbal.y)
                    sx = 0.0 if abs(sx) < DEADBAND_DEG else max(-MAX_STEP_DEG, min(MAX_STEP_DEG, sx))
                    sy = 0.0 if abs(sy) < DEADBAND_DEG else max(-MAX_STEP_DEG, min(MAX_STEP_DEG, sy))
                    if sx or sy:
                        gimbal.move_by(sx, sy)

                line = (f"{state:6} dx={dx:+7.1f}px dy={dy:+7.1f}px dist={math.hypot(dx, dy):6.1f}px "
                        f"({off_m:4.2f} m) disc={det['w']:.0f}x{det['h']:.0f}px tilt={det['tilt']:4.1f}deg "
                        f"[{det['source']}] gimbal=({gimbal.x:+5.1f},{gimbal.y:+5.1f})deg")
                if state == "LOCKED" and estimates:
                    a_ = np.array(estimates)
                    mx, my = a_.mean(axis=0)
                    result = (mx, my)
                    line += f" | TARGET field=({mx:6.2f}, {my:6.2f}) m +-{float(np.hypot(*a_.std(axis=0))):.2f}"

            # quad / circle geometry logging (Naklon)
            geometry_frame_counter += 1
            if args.show or args.geometry:
                if geometry is None or geometry_frame_counter % GEOMETRY_EVERY_N_FRAMES == 0:
                    geometry = analyze_object_geometry(frame, prefer=last_pos if det is not None else None)
                visible_circles = len(geometry["circles"]) if geometry else 0
                object_angle = geometry["angle"] if geometry else None
                if geometry and geometry["circles"]:
                    log_center = geometry["circles"][0]["center"]
                elif geometry:
                    log_center = geometry["center"]
                else:
                    log_center = last_pos if last_pos is not None else (W / 2.0, H / 2.0)
                count_changed = last_logged_circle_count is None or visible_circles != last_logged_circle_count
                angle_changed = object_angle is not None and (
                    last_logged_angle is None or abs(object_angle - last_logged_angle) > ANGLE_LOG_DELTA_DEG)
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
            if now - t_print >= PRINT_EVERY_S:
                print(f"{line} | {fps:4.1f} fps")
                t_print = now

            if args.show:
                ds = float(DISPLAY_SCALE)
                view = cv2.resize(frame, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
                vH, vW = view.shape[:2]
                colour = {"LOCKED": (0, 255, 0), "TRACK": (0, 255, 255)}.get(state, (200, 200, 200))
                cv2.drawMarker(view, (vW // 2, vH // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                if det is not None:
                    c = (int(round(det["x"] * ds)), int(round(det["y"] * ds)))
                    axes = (max(3, int(det["w"] * ds / 2)),
                            max(2, int(det["h"] * ds / 2)))

                    # Restore the old behaviour: the actually detected circular/
                    # elliptical target is visibly filled red in the preview.
                    cv2.ellipse(view, c, axes, det["angle"], 0, 360,
                                (0, 0, 255), -1)
                    cv2.ellipse(view, c, axes, det["angle"], 0, 360,
                                (255, 255, 255), 2)
                    cv2.line(view, (vW // 2, vH // 2), c, colour, 1)
                if geometry is not None:
                    q = np.rint(geometry["contour"].astype(np.float32) * ds).astype(np.int32)
                    cv2.drawContours(view, [q], -1, (255, 0, 255), 1)
                cv2.putText(view, state, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                # A tracked round target counts as visible even when there is
                # no surrounding quadrilateral in the current bench test.
                shown_circle_count = len(geometry["circles"]) if geometry is not None else 0
                if det is not None:
                    shown_circle_count = max(1, shown_circle_count)
                tilt_text = (
                    f"{geometry['angle']:+.1f}"
                    if geometry is not None and geometry["angle"] is not None
                    else "N/A"
                )
                cv2.putText(view,
                            f"Viditelne kruhy: {shown_circle_count}  Naklon: {tilt_text}",
                            (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
                if det is not None:
                    cv2.putText(view, f"disc tilt {det['tilt']:.0f} deg [{det['source']}]", (10, 74),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(view, f"FPS: {fps:.1f}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("tracker (q = quit)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds and now - t_start > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        gimbal.close()
        cam.close()
        _close_windows()
    if t_first_lock:
        print(f"[summary] first LOCK after {t_first_lock - t_start:.1f} s")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="preview window")
    ap.add_argument("--no-servo", action="store_true", help="don't move the gimbal (detection test)")
    ap.add_argument("--bench", type=float, default=0, metavar="DIST_M",
                    help="desk test: camera-to-target distance in metres, no search sweep")
    ap.add_argument("--geometry", action="store_true", help="print quad/circle (Naklon) info without --show")
    ap.add_argument("--tune", action="store_true", help="colour threshold sliders")
    ap.add_argument("--sim", nargs=2, type=float, metavar=("X", "Y"), help="simulate, target at field X, Y")
    ap.add_argument("--tilt", type=float, default=0.0, help="sim: tilt of the printed disc (deg)")
    ap.add_argument("--pink", action="store_true", help="sim: NoIR pink rendering of the red ink")
    ap.add_argument("--glare", action="store_true", help="sim: glare spot on the disc")
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds")
    run(ap.parse_args())
