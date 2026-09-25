#!/usr/bin/env python3
"""
tune_hsv.py - interactively find HSV thresholds for the red target.

Run on a PC (or the Pi with a desktop) on frames captured with
    python3 red_circle_tracker.py --save-raw 20
then:
    python3 tune_hsv.py raw_frames/frame_0005.png

Move the sliders until only the target is white in the mask.
Click on the target to print its HSV value. Press q to quit;
the matching command-line flags are printed.
"""
import sys
import cv2
import numpy as np

path = sys.argv[1]
img = cv2.imread(path)
if img is None:
    sys.exit(f"cannot read {path}")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
scale = min(1.0, 1100.0 / img.shape[1])

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
for name, val, mx in [("h_low", 10, 90), ("h_high", 160, 180),
                      ("s_min", 90, 255), ("v_min", 60, 255)]:
    cv2.createTrackbar(name, "controls", val, mx, lambda x: None)


def on_click(event, x, y, flags, _):
    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = int(x / scale), int(y / scale)
        patch = hsv[max(0, Y - 2):Y + 3, max(0, X - 2):X + 3].reshape(-1, 3)
        print(f"pixel ({X},{Y}) HSV median = {np.median(patch, axis=0).astype(int).tolist()}")


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_click)
while True:
    hl = cv2.getTrackbarPos("h_low", "controls")
    hh = cv2.getTrackbarPos("h_high", "controls")
    sm = cv2.getTrackbarPos("s_min", "controls")
    vm = cv2.getTrackbarPos("v_min", "controls")
    m = cv2.inRange(hsv, (0, sm, vm), (hl, 255, 255)) | cv2.inRange(hsv, (hh, sm, vm), (180, 255, 255))
    cv2.imshow("image", cv2.resize(img, None, fx=scale, fy=scale))
    cv2.imshow("mask", cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(30) & 0xFF in (27, ord("q")):
        break
print(f"--h-low {hl} --h-high {hh} --s-min {sm} --v-min {vm}")
cv2.destroyAllWindows()
