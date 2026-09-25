#!/usr/bin/env python3
"""
Gimbal setup - run this BEFORE red_tracker.py
==============================================
STEP 1  Servo range (no camera / disc needed)
        You move each servo with the keyboard in small steps and mark where it
        reaches its end stops. The tracker will never drive past these points.
        (Driving into an end stop = servo pushes forever, draws big current,
        the Pi/camera brown out and the tracker never reaches the target.)

STEP 2  Directions (automatic)
        The gimbal stays still until the red disc is seen - put it in front of the
        camera whenever you like, anywhere in the picture (no need to centre it).
        Then each servo is nudged a little and the script watches which way the disc
        moves, and works out if the servos are swapped / reversed.

Everything is saved to gimbal_calibration.json, which red_tracker.py loads.

Keys (in the terminal, or in the preview window with --show):
    a / d     servo -5 / +5 us        A / D   -50 / +50 us
    m         mark current position as an END (just before it touches the stop)
    c         mark current position as CENTRE (optional; default = middle of the ends)
    o         servo off (stops buzzing)          Enter / n   next
    q         quit without saving

Run on the Pi:
    python3 gimbal_setup.py --show        # with camera preview (desktop / VNC)
    python3 gimbal_setup.py               # over SSH, no window
    python3 gimbal_setup.py --skip-range  # only redo the directions
"""

import argparse
import json
import os
import select
import sys
import time

import cv2
import numpy as np

import red_tracker as rt

WIN = "gimbal setup (keys here)"
END_MARGIN_US = 40          # keep this far away from each marked end stop
STABLE_FRAMES = 6           # disc must be seen this many frames in a row before moving


# ------------------------------------------------------------------ keyboard + preview
class Console:
    def __init__(self, show, cam):
        self.show, self.cam = show, cam
        self.lines, self.det = [], None
        self.tty = sys.stdin.isatty() and not show
        if self.tty:
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def status(self, *lines):
        if list(lines) != self.lines:
            self.lines = list(lines)
            print("\n".join(lines))

    def key(self, timeout=0.05):
        """One key press or None. Also refreshes the preview window."""
        if self.show:
            self.frame()
            k = cv2.waitKey(max(1, int(timeout * 1000))) & 0xFF
            return None if k == 255 else ("\n" if k in (10, 13) else chr(k))
        if self.tty:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            return sys.stdin.read(1) if r else None
        # piped input (testing): one character at a time
        ch = sys.stdin.read(1)
        if not ch:
            raise KeyboardInterrupt
        return ch

    def poll(self):
        """Non-blocking key check that never consumes piped test input."""
        if self.show:
            k = cv2.waitKey(1) & 0xFF
            return None if k == 255 else chr(k)
        if self.tty:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            return sys.stdin.read(1) if r else None
        return None

    def frame(self):
        """Grab + (optionally) show a camera frame. Returns the frame or None."""
        if self.cam is None:
            return None
        f = self.cam.read()
        if f is None:
            return None
        if self.show:
            ds = rt.DISPLAY_SCALE
            v = cv2.resize(f, None, fx=ds, fy=ds)
            if self.det is not None:
                c = (int(self.det["x"] * ds), int(self.det["y"] * ds))
                cv2.circle(v, c, max(4, int(self.det["w"] * ds / 2)), (0, 255, 0), 2)
            for i, t in enumerate(self.lines[-4:]):
                y = 24 + 22 * i
                cv2.putText(v, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(v, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow(WIN, v)
        return f

    def close(self):
        if self.tty:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
        if self.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ------------------------------------------------------------------ servo output
class Servos:
    def __init__(self, dry):
        self.pi = None
        if not dry:
            import pigpio
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpiod not running: sudo systemctl start pigpiod")

    def set(self, pin, us):
        if self.pi:
            self.pi.set_servo_pulsewidth(pin, 0 if us is None else int(us))

    def off(self):
        for p in (rt.X_PIN, rt.Y_PIN):
            self.set(p, None)

    def close(self):
        if self.pi:
            self.off()
            self.pi.stop()
            self.pi = None


# ------------------------------------------------------------------ step 1: range
def setup_range(con, servos, pin, name, start_us, other=None):
    """Jog one servo, mark both end stops. Returns (min_us, center_us, max_us) or None."""
    if other:                                   # hold the other servo at its centre
        servos.set(*other)
    us, on, ends, centre = int(start_us), True, [], None
    servos.set(pin, us)
    print(f"\n=== STEP 1: {name} (pin {pin}) range ===")
    print("Move it slowly with a/d (A/D = big steps) towards ONE end until it just stops")
    print("moving (or starts to buzz), then back off a little and press m.")
    print("Then do the same at the OTHER end. c = mark the straight/level position (optional).")
    print("o = servo off,  Enter = done,  q = quit\n")
    while True:
        marks = ", ".join(str(e) for e in ends) or "-"
        con.status(f"{name}: {us} us {'' if on else '(OFF)'}  ends: {marks}"
                   f"  centre: {centre or 'auto'}")
        k = con.key()
        if k is None:
            continue
        step = {"a": -5, "d": 5, "A": -50, "D": 50}.get(k)
        if step:
            us, on = max(500, min(2500, us + step)), True
            servos.set(pin, us)
        elif k == "m":
            ends.append(us)
            ends = ends[-2:]
            print(f"  end marked at {us} us")
        elif k == "c":
            centre = us
            print(f"  centre marked at {us} us")
        elif k == "o":
            on = False
            servos.set(pin, None)
        elif k == "q":
            return None
        elif k in ("\n", "\r", "n"):
            if len(ends) < 2 or abs(ends[0] - ends[1]) < 100:
                print("  mark BOTH ends first (m at each end)")
                continue
            lo, hi = min(ends) + END_MARGIN_US, max(ends) - END_MARGIN_US
            c = int(round((lo + hi) / 2)) if centre is None else max(lo, min(hi, centre))
            servos.set(pin, c)
            print(f"  {name}: {lo} .. {c} .. {hi} us  "
                  f"(about {(hi - lo) / rt.US_PER_DEG:.0f} deg of travel)")
            return lo, c, hi


# ------------------------------------------------------------------ step 2: directions
def wait_for_disc(con):
    """Gimbal stays still. Returns a stable detection."""
    print("\n=== STEP 2: directions ===")
    print("Waiting for the red disc - hold it in front of the camera, anywhere in the picture,")
    print("and keep it STILL for ~10 s once it is seen.  (q = quit)\n")
    run, last = 0, None
    while True:
        frame = con.frame()
        if con.poll() == "q":
            return None
        if frame is None:
            continue
        rt.COLOUR.update(frame)
        det, _ = rt.detect(frame, last["w"] if last else 100.0,
                           prefer=(last["x"], last["y"]) if last else None,
                           size_range=(0.5, 2.0) if last else (0.1, 6.0))
        con.det = det
        still = (det is not None and last is not None and
                 np.hypot(det["x"] - last["x"], det["y"] - last["y"]) < max(6.0, 0.1 * det["w"]))
        run = run + 1 if still else (1 if det else 0)
        last = det
        con.status("waiting for the red disc ..." if det is None else
                   f"disc seen ({run}/{STABLE_FRAMES}) - keep it still")
        if det is not None:
            rt.COLOUR.learn(det.get("score"))
        if run >= STABLE_FRAMES:
            return det


def setup_directions(con, cam):
    det = wait_for_disc(con)
    if det is None:
        return None
    con.status("disc found - nudging the servos, keep the disc still ...")
    gimbal = rt.Gimbal(hardware=False)          # only for bookkeeping; pulses go via pigpio below
    gimbal._write = lambda: _write_pulses(gimbal)
    gimbal._write()
    P = rt.calibrate_gimbal(cam, gimbal, det["w"], (det["x"], det["y"]))
    return P


_SERVOS = None


def _write_pulses(g):
    px = rt.X_CENTER_US + rt.X_DIR * g.cx * rt.US_PER_DEG
    py = rt.Y_CENTER_US + rt.Y_DIR * g.cy * rt.US_PER_DEG
    _SERVOS.set(rt.X_PIN, max(rt.X_MIN_US, min(rt.X_MAX_US, px)))
    _SERVOS.set(rt.Y_PIN, max(rt.Y_MIN_US, min(rt.Y_MAX_US, py)))


# ------------------------------------------------------------------ main
def load_existing(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def main():
    global _SERVOS
    ap = argparse.ArgumentParser(description="Measure servo ranges and directions.")
    ap.add_argument("--show", action="store_true", help="camera preview window (keys go there)")
    ap.add_argument("--skip-range", action="store_true", help="keep the saved servo ranges")
    ap.add_argument("--skip-directions", action="store_true", help="only measure the ranges")
    ap.add_argument("--out", default=rt.CALIB_FILE)
    ap.add_argument("--dry-run", action="store_true", help="no servos (test the script)")
    ap.add_argument("--sim", action="store_true", help="simulated camera + gimbal (testing)")
    ap.add_argument("--sim-swap", action="store_true")
    ap.add_argument("--sim-invert-x", action="store_true")
    args = ap.parse_args()

    data = load_existing(args.out)
    _SERVOS = Servos(dry=args.dry_run or args.sim)
    cam = con = None
    try:
        # ---------------- step 1
        if args.skip_range:
            if not rt.load_servo_ranges(args.out):
                print("No saved ranges - run without --skip-range first.")
                return 1
        else:
            _SERVOS.off()
            con = Console(False, None)          # step 1 keys go in the terminal
            rx = setup_range(con, _SERVOS, rt.X_PIN, "X servo", rt.X_CENTER_US)
            if rx is None:
                print("Quit - nothing saved.")
                return 0
            ry = setup_range(con, _SERVOS, rt.Y_PIN, "Y servo", rt.Y_CENTER_US,
                             other=(rt.X_PIN, rx[1]))
            if ry is None:
                print("Quit - nothing saved.")
                return 0
            con.close()
            con = None
            rt.X_MIN_US, rt.X_CENTER_US, rt.X_MAX_US = rx
            rt.Y_MIN_US, rt.Y_CENTER_US, rt.Y_MAX_US = ry
            data["servo"] = {"x": dict(min_us=rx[0], center_us=rx[1], max_us=rx[2]),
                             "y": dict(min_us=ry[0], center_us=ry[1], max_us=ry[2])}
            data.pop("P", None)                 # new ranges -> directions must be redone
            _save(data, args.out)

        # ---------------- step 2
        if not args.skip_directions:
            if args.sim:
                rt.DRONE_ALT_M = 1.5
                cam = DelayedSim(args)
            else:
                cam = rt.Camera()
            con = Console(args.show, cam)
            P = setup_directions(con, cam)
            if P is None:
                print("\nDirections NOT measured (quit, or the disc was lost). "
                      "Ranges are saved; run again with --skip-range.")
                return 1
            data["P"] = np.asarray(P).tolist()
            _save(data, args.out)
        print("\nDone. Now run:  python3 red_tracker.py --bench 1.5 --show")
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    finally:
        if con:
            con.close()
        if cam:
            cam.close()
        _SERVOS.close()


def _save(data, path):
    data.update(saved=time.strftime("%Y-%m-%d %H:%M:%S"), source="manual",
                note="made by gimbal_setup.py; physical(right, forward) = P @ raw servo command")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {os.path.basename(path)}")


class DelayedSim:
    """Test only: simulated gimbal camera; the disc appears after 3 s."""

    def __init__(self, args):
        self.g = rt.Gimbal(hardware=False)
        hw = np.eye(2)
        if args.sim_swap:
            hw = hw[::-1]
        if args.sim_invert_x:
            hw = hw @ np.diag([-1.0, 1.0])
        self.sim = rt.SimCamera(self.g, (30.3, 15.2), hw_mapping=hw)
        self.t0 = time.time()
        self._pulses = None
        global _write_pulses
        orig = _write_pulses

        def track(g):             # mirror the commands into the sim gimbal
            orig(g)
            self.g.cx, self.g.cy = g.cx, g.cy
            self.g.hist.append((time.time(), 0.0, 0.0, g.cx, g.cy))
        _write_pulses = track

    def read(self):
        f = self.sim.read()
        if time.time() - self.t0 < 3.0:
            return cv2.GaussianBlur(self.sim.bg, (3, 3), 0)
        return f

    def close(self):
        pass


if __name__ == "__main__":
    sys.exit(main())
