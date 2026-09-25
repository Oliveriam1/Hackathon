#!/usr/bin/env python3
"""
Gimbal setup - run this BEFORE red_tracker.py
==============================================
Servos: two 360-degree positional servos on physical pins 38 (GPIO20, "servo A")
and 40 (GPIO21, "servo B"). Settings come from red_tracker.py.

STEP 1  Servo check   - each servo wiggles on its own; you say if it moved.
                        (Finds a dead servo / wrong pin / missing power straight away.)
STEP 2  Servo range   - you move each servo with the keyboard and mark where it reaches
                        its end stops. The tracker will never drive past these points.
STEP 3  Directions    - automatic. The gimbal stays still until the red disc is seen
                        (anywhere in the picture, no need to centre it), then nudges
                        each servo and works out which one turns the camera left/right,
                        which one up/down, and which way round.

Everything is saved to gimbal_calibration.json, which red_tracker.py loads.

Keys for steps 1-2 go in the TERMINAL:
    a / d   -1 / +1 deg          A / D   -10 / +10 deg
    m       mark an END (just before it touches the stop)
    c       mark the CENTRE (optional; default = middle between the ends)
    o       servo off (stops buzzing)       Enter   done        q   quit

Run on the Pi:
    python3 gimbal_setup.py --show          # preview window for step 3 (desktop / VNC)
    python3 gimbal_setup.py                 # over SSH, no window
    python3 gimbal_setup.py --skip-range    # only redo the directions (step 3)
"""

import argparse
import select
import sys
import time

import cv2
import numpy as np

import red_tracker as rt

END_MARGIN_DEG = 4.0        # stay this far inside each marked end stop
WIGGLE_DEG = 8.0            # step 1 test movement
STABLE_FRAMES = 6           # disc seen this many frames in a row (still) before moving


# ------------------------------------------------------------------ keyboard
class Keys:
    """Single key presses from the terminal (works over SSH); piped input for testing."""

    def __init__(self):
        self.tty = sys.stdin.isatty()
        if self.tty:
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def get(self, timeout=0.1):
        if self.tty:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            return sys.stdin.read(1) if r else None
        ch = sys.stdin.read(1)
        if not ch:
            raise KeyboardInterrupt
        return ch

    def poll(self):
        if self.tty:
            r, _, _ = select.select([sys.stdin], [], [], 0)
            return sys.stdin.read(1) if r else None
        return None

    def ask(self, keys):
        while True:
            k = self.get(0.2)
            if k and k.lower() in keys:
                return k.lower()

    def close(self):
        if self.tty:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)


def status(text):
    sys.stdout.write("\r" + text.ljust(90))
    sys.stdout.flush()


# ------------------------------------------------------------------ step 1: servo check
def servo_check(keys, servos):
    print("\n=== STEP 1: servo check ===")
    ok = True
    for sv in servos:
        others = [s for s in servos if s is not sv]
        while True:
            for o in others:
                o.off()
            print(f"\n{sv.name} will now move a little back and forth 3 times - watch the gimbal.")
            sv.set(0.0)
            time.sleep(0.8)
            for _ in range(3):
                for a in (WIGGLE_DEG, -WIGGLE_DEG):
                    sv.set(a)
                    time.sleep(0.5)
            sv.set(0.0)
            time.sleep(0.4)
            sv.off()
            print("Did it move?  y = yes,  n = no,  r = repeat")
            k = keys.ask("ynr")
            if k == "r":
                continue
            if k == "n":
                ok = False
                print(f"  -> {sv.name} did NOT move: check its 5 V supply, the GROUND shared with the "
                      f"Pi, and the signal wire on physical pin {rt.BCM_TO_PHYS.get(sv.pin)} "
                      f"(GPIO{sv.pin}).")
            break
    return ok


# ------------------------------------------------------------------ step 2: range
def servo_range(keys, sv, start_deg=0.0):
    """Jog one servo, mark both end stops. Returns (center_us, min_us, max_us) or None."""
    print(f"\n=== STEP 2: {sv.name} range ===")
    print("Move it with a/d (A/D = big steps) slowly towards ONE end until it stops moving")
    print("(or buzzes), back off a little and press m. Then the same at the OTHER end.")
    print("c = mark the straight / level position (optional),  o = off,  Enter = done,  q = quit\n")
    # open the range wide for jogging (the whole electrical range of the servo)
    sv.min_us, sv.max_us = rt.SERVO_PULSE_MIN_US, rt.SERVO_PULSE_MAX_US
    ang, on, ends, centre = start_deg, True, [], None
    sv.set(ang)
    while True:
        marks = ", ".join(f"{e:+.0f}" for e in ends) or "-"
        status(f"{sv.name}: {ang:+6.1f} deg ({sv.pulse()} us){'' if on else ' OFF'}   "
               f"ends: {marks}   centre: {'auto' if centre is None else f'{centre:+.0f}'}")
        k = keys.get()
        if k is None:
            continue
        step = {"a": -1.0, "d": 1.0, "A": -10.0, "D": 10.0}.get(k)
        if step:
            ang, on = ang + step, True
            sv.set(ang)
            ang = sv.angle
        elif k == "m":
            ends = (ends + [ang])[-2:]
            print(f"\n  end marked at {ang:+.0f} deg")
        elif k == "c":
            centre = ang
            print(f"\n  centre marked at {ang:+.0f} deg")
        elif k == "o":
            on = False
            sv.off()
        elif k == "q":
            print()
            return None
        elif k in ("\n", "\r"):
            if len(ends) < 2 or abs(ends[0] - ends[1]) < 2 * END_MARGIN_DEG + 4:
                print("\n  mark BOTH ends first (m at each end)")
                continue
            lo, hi = min(ends) + END_MARGIN_DEG, max(ends) - END_MARGIN_DEG
            c = (lo + hi) / 2 if centre is None else max(lo, min(hi, centre))
            to_us = lambda a: int(round(sv.center_us + a * rt.US_PER_DEG))
            result = (to_us(c), to_us(lo), to_us(hi))
            print(f"\n  {sv.name}: {lo - c:+.0f} .. {hi - c:+.0f} deg around the centre "
                  f"({result[1]}..{result[0]}..{result[2]} us)")
            sv.center_us, sv.min_us, sv.max_us = result
            sv.angle = 0.0
            sv.set(0.0)                          # park at the new centre
            return result


# ------------------------------------------------------------------ step 3: directions
def wait_for_disc(cam, keys, show):
    print("\n=== STEP 3: directions ===")
    print("Waiting for the red disc - hold it in front of the camera, anywhere in the picture,")
    print("and keep it STILL for ~10 s once it is seen.  (q = quit)\n")
    run, last = 0, None
    while True:
        if keys.poll() == "q":
            return None
        frame = cam.read()
        if frame is None:
            continue
        rt.COLOUR.update(frame)
        det, _ = rt.detect(frame, last["w"] if last else 100.0,
                           prefer=(last["x"], last["y"]) if last else None,
                           size_range=(0.5, 2.0) if last else (0.1, 6.0))
        still = det is not None and last is not None and \
            np.hypot(det["x"] - last["x"], det["y"] - last["y"]) < max(6.0, 0.1 * det["w"])
        run = run + 1 if still else (1 if det else 0)
        last = det
        if det is not None:
            rt.COLOUR.learn(det.get("score"))
        status("waiting for the red disc ..." if det is None else
               f"disc seen ({run}/{STABLE_FRAMES}) - keep it still")
        if show:
            v = cv2.resize(frame, None, fx=rt.DISPLAY_SCALE, fy=rt.DISPLAY_SCALE)
            if det is not None:
                c = (int(det["x"] * rt.DISPLAY_SCALE), int(det["y"] * rt.DISPLAY_SCALE))
                cv2.circle(v, c, max(4, int(det["w"] * rt.DISPLAY_SCALE / 2)), (0, 255, 0), 2)
            cv2.imshow("gimbal setup", v)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None
        if run >= STABLE_FRAMES:
            print()
            return det


class SimSource:
    """Testing only: simulated camera, the disc appears after 3 s."""

    def __init__(self, gimbal, args):
        rt.DRONE_ALT_M = 1.5
        hw = np.eye(2)
        if args.sim_swap:
            hw = hw[::-1].copy()
        if args.sim_invert_x:
            hw[0] *= -1
        self.sim = rt.SimCamera(gimbal, (30.3, 15.2), hw_mapping=hw)
        self.t0 = time.time()

    def read(self):
        f = self.sim.read()
        return cv2.GaussianBlur(self.sim.bg, (3, 3), 0) if time.time() - self.t0 < 3 else f

    def close(self):
        pass


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Check the servos, measure their range and directions.")
    ap.add_argument("--show", action="store_true", help="camera preview window in step 3")
    ap.add_argument("--skip-range", action="store_true", help="keep the saved ranges, only step 3")
    ap.add_argument("--skip-directions", action="store_true", help="only steps 1-2")
    ap.add_argument("--sim", action="store_true", help="test without hardware")
    ap.add_argument("--sim-swap", action="store_true")
    ap.add_argument("--sim-invert-x", action="store_true")
    args = ap.parse_args()

    pi = None
    if not args.sim:
        import pigpio
        pi = pigpio.pi()
        if not pi.connected:
            print("pigpiod is not running:  sudo systemctl start pigpiod")
            return 1
    keys = Keys()
    cfg = rt.load_gimbal_config(verbose=False)
    gimbal, cam = None, None
    try:
        if not args.skip_range:
            servos = [rt.Servo(i, rt.SERVO_PINS[i], rt.DEFAULT_CENTER_US,
                               rt.SERVO_PULSE_MIN_US, rt.SERVO_PULSE_MAX_US, pi=pi) for i in range(2)]
            if not servo_check(keys, servos):
                print("\nFix the servo(s) that did not move, then run this again.")
                return 1
            ranges = []
            for sv in servos:
                for o in servos:
                    if o is not sv:
                        o.off()
                r = servo_range(keys, sv)
                if r is None:
                    print("Quit - nothing saved.")
                    return 0
                ranges.append(r)
            rt.save_gimbal_config(servos=ranges, source="gimbal_setup.py")
            cfg = rt.load_gimbal_config(verbose=True)
        elif not cfg["measured_range"]:
            print("No measured ranges saved yet - run without --skip-range first.")
            return 1

        if args.skip_directions:
            return 0
        cfg["axes"] = None
        gimbal = rt.Gimbal(hardware=False, cfg=cfg)
        for s in gimbal.servos:                   # drive the real servos through our pigpio link
            s.pi = pi
            s.set(0.0)
        cam = SimSource(gimbal, args) if args.sim else rt.Camera()
        det = wait_for_disc(cam, keys, args.show)
        if det is None:
            print("\nQuit - directions not measured (ranges are saved).")
            return 1
        axes = rt.calibrate_gimbal(cam, gimbal, det["w"], (det["x"], det["y"]))
        if axes is None:
            print("\nDirections NOT measured - see the message above. "
                  "Run again with --skip-range to retry.")
            return 1
        rt.save_gimbal_config(gimbal, source="gimbal_setup.py")
        print("\nDone. Now run:  python3 red_tracker.py --bench 1.5 --show")
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 1
    finally:
        keys.close()
        if cam:
            cam.close()
        if pi:
            for p in rt.SERVO_PINS:
                pi.set_servo_pulsewidth(p, 0)
            pi.stop()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    sys.exit(main())
