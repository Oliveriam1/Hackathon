#!/usr/bin/env python3
"""
Gimbal direction setup - run this ONCE before using red_tracker.py
===================================================================
Moves each servo on its own and asks you which way the picture moved.
From your answers it works out whether the servos are swapped and/or reversed,
lets you check the result with up/down/left/right keys, and saves it to
gimbal_calibration.json. red_tracker.py loads that file automatically.

All answers are about the CAMERA PICTURE on the screen (not the drone, not the
servo arm). That is what the tracker works with, so a sideways-mounted camera
is fine.

Put something easy to see (the red disc) in the middle of the picture first.

Run on the Pi:
    python3 gimbal_setup.py --show          # with camera preview window (desktop / VNC)
    python3 gimbal_setup.py                 # terminal only: watch the picture another way
    python3 gimbal_setup.py --step 20       # bigger test moves
    python3 gimbal_setup.py --dry-run       # no servos, just try the questions
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

import red_tracker as rt

WIN = "gimbal setup (keys in this window)"


# ------------------------------------------------------------------ UI
class UI:
    """Asks questions either in the preview window (--show) or in the terminal."""

    def __init__(self, show):
        self.cam = rt.Camera() if show else None
        self.lines = []

    # --- preview ---
    def _frame(self):
        frame = self.cam.read()
        if frame is None:
            frame = np.zeros((rt.CAPTURE_H, rt.CAPTURE_W, 3), np.uint8)
        view = cv2.resize(frame, None, fx=rt.DISPLAY_SCALE, fy=rt.DISPLAY_SCALE)
        h, w = view.shape[:2]
        cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
        for i, t in enumerate(self.lines):
            y = 26 + 24 * i
            cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(view, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WIN, view)
        return cv2.waitKey(1) & 0xFF

    def say(self, *lines):
        self.lines = list(lines)
        for t in lines:
            print(t)

    def wait(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            if self.cam:
                self._frame()
            else:
                time.sleep(0.05)

    def ask(self, lines, keys):
        """Show the question, return one of `keys` (single characters)."""
        self.say(*lines)
        if self.cam:
            while True:
                k = self._frame()
                if k != 255 and chr(k).lower() in keys:
                    return chr(k).lower()
        while True:
            a = input(f"  answer [{'/'.join(keys)}] + Enter: ").strip().lower()
            if a[:1] in keys:
                return a[:1]

    def close(self):
        if self.cam:
            self.cam.close()
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ------------------------------------------------------------------ logic
# answer -> column of P (physical deg per +1 raw deg).
# Tracker convention: physical +x = the OBJECT moves LEFT on screen,
#                     physical +y = the OBJECT moves DOWN on screen.
ANSWER_TO_COL = {"l": (1.0, 0.0), "r": (-1.0, 0.0), "d": (0.0, 1.0), "u": (0.0, -1.0)}
WORD = {"l": "LEFT", "r": "RIGHT", "u": "UP", "d": "DOWN"}


def test_servo(ui, gimbal, axis, step):
    name = f"{'X' if axis == 0 else 'Y'} servo (pin {rt.X_PIN if axis == 0 else rt.Y_PIN})"
    while True:
        gimbal.set_raw(0.0, 0.0)
        ui.say(f"Testing {name}", "Centre - watch the object in the picture...")
        ui.wait(1.5)
        for n in (3, 2, 1):
            ui.say(f"Testing {name}", f"Moving in {n} ...")
            ui.wait(0.6)
        raw = [0.0, 0.0]
        raw[axis] = step
        gimbal.set_raw(*raw)
        ui.wait(0.8)
        a = ui.ask([f"{name} moved +{step:.0f} deg.",
                    "Which way did the OBJECT move on the screen?",
                    "l=left  r=right  u=up  d=down",
                    "n=did not move   a=show again"], "lrudna")
        if a == "a":
            continue
        gimbal.set_raw(0.0, 0.0)
        if a == "n":
            return name, None
        print(f"  -> {name}: object moves {WORD[a]}")
        return name, a


def jog_check(ui, gimbal, step):
    """Move by picture directions. Returns 'y' (save), 'r' (redo) or 'q' (quit)."""
    gimbal.move_to(0.0, 0.0)
    moves = {"a": (1.0, 0.0), "d": (-1.0, 0.0), "s": (0.0, 1.0), "w": (0.0, -1.0)}
    while True:
        k = ui.ask(["CHECK: press a key - the OBJECT should move that way on screen",
                    "w=up  s=down  a=left  d=right   c=centre",
                    f"(now x={gimbal.x:+.0f}, y={gimbal.y:+.0f} deg)",
                    "y=correct, SAVE   r=wrong, redo test   q=quit"], "wasdcyrq")
        if k in moves:
            dx, dy = moves[k]
            gimbal.move_by(dx * step, dy * step)
        elif k == "c":
            gimbal.move_to(0.0, 0.0)
        else:
            return k


def describe(P):
    straight = P[0, 1] == 0
    xs, ys = (P[0, 0], P[1, 1]) if straight else (P[1, 0], P[0, 1])
    problems = []
    if not straight:
        problems.append("X and Y servos are SWAPPED")
    if xs < 0:
        problems.append("X servo direction is REVERSED")
    if ys < 0:
        problems.append("Y servo direction is REVERSED")
    return "; ".join(problems) if problems else "wired as expected (nothing to fix)"


def save(P, path):
    with open(path, "w") as f:
        json.dump({"P": P.tolist(),
                   "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "source": "manual",
                   "note": "physical(right, forward) = P @ raw servo command (X_PIN, Y_PIN); "
                           "made by gimbal_setup.py"}, f, indent=2)
    print(f"\nSaved to {path}")


def main():
    ap = argparse.ArgumentParser(description="Find and fix the gimbal servo directions.")
    ap.add_argument("--show", action="store_true", help="camera preview window with the questions")
    ap.add_argument("--step", type=float, default=15.0, help="test move in degrees (default 15)")
    ap.add_argument("--dry-run", action="store_true", help="don't drive the servos")
    ap.add_argument("--out", default=rt.CALIB_FILE, help="output file")
    args = ap.parse_args()

    gimbal = rt.Gimbal(hardware=not args.dry_run)     # identity mapping = raw commands
    ui = UI(args.show)
    try:
        print("Put the red disc (or any clear object) in the MIDDLE of the camera picture.\n")
        while True:
            gimbal.set_mapping(np.eye(2))
            gimbal.set_raw(0.0, 0.0)
            answers = []
            for axis in (0, 1):
                name, a = test_servo(ui, gimbal, axis, args.step)
                if a is None:
                    print(f"\n{name}: the picture did not move. Check that servo's power, "
                          f"ground and signal wire (BCM pin), then run again.")
                    return 1
                answers.append(a)

            cols = [np.array(ANSWER_TO_COL[a]) for a in answers]
            if abs(cols[0] @ cols[1]) > 0:     # both along the same screen axis
                ui.ask([f"Both servos moved the object {WORD[answers[0]]}/{WORD[answers[1]]}",
                        "(same axis) - that can't be right.", "press r to redo"], "r")
                continue

            P = np.column_stack(cols)
            print(f"\nResult: {describe(P)}")
            print(f"P = {P.tolist()}\n")
            gimbal.set_mapping(P)
            k = jog_check(ui, gimbal, max(3.0, args.step / 3))
            if k == "y":
                save(P, args.out)
                print("red_tracker.py will now use these directions "
                      "(and will not overwrite them automatically).")
                return 0
            if k == "q":
                print("Quit - nothing saved.")
                return 0
            print("\nRedoing the test...\n")
    except KeyboardInterrupt:
        print("\nStopped - nothing saved.")
        return 1
    finally:
        gimbal.move_to(0.0, 0.0)
        time.sleep(0.3)
        gimbal.close()
        ui.close()


if __name__ == "__main__":
    sys.exit(main())
