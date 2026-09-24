# Hackathon – IR circle in quadrilateral

Minimal version that does one thing: it finds a circle enclosed by a rectangle / four-sided marker.
It does **not** use colour, so a strong red/magenta cast from an infrared-sensitive Raspberry Pi camera does not affect the classification logic.

## Raspberry Pi setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

Check the camera first:

```bash
rpicam-hello --list-cameras
rpicam-hello -t 5000
```

## Run

Live CSI camera:

```bash
python main.py
```

Headless / SSH – capture one annotated frame:

```bash
python main.py --snapshot test.jpg
```

Test without hardware:

```bash
python main.py --demo --snapshot demo.jpg
```

Test an existing image:

```bash
python main.py --image photo.jpg --snapshot result.jpg
```

## What the detector does

1. Converts the camera frame to intensity only (no colour classification).
2. Detects four-sided contours.
3. Rectifies each quadrilateral with a perspective transform.
4. Looks for a circular/elliptic contour inside the rectified region.
5. Maps the detected circle centre back to the original camera image.

The output currently contains only the circle centre, approximate radius, enclosing quadrilateral and confidence.
