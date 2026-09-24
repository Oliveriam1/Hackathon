# Hackathon – circle detection and mission simulation

The camera mode finds a circle enclosed by a rectangle / four-sided marker.
The new mission mode plans a route, computes scan height, detects the circle,
maps it to ground coordinates and reports JSON. The supplied flight adapter is
**simulation only**: no commands reach UAV hardware.

The detector uses geometry/intensity rather than colour labels. This avoids
depending on red/green hues from a NoIR camera, but does not reconstruct missing
visible-spectrum information or guarantee detection in every infrared scene.
Pi capture explicitly enables automatic exposure/white balance and uses BGR
output (`RGB888` in Picamera2), without an extra channel swap.

## Live video in a browser

No GUI window or additional web framework is needed. Test on a generated scene:

```bash
python main.py --demo --stream
```

Open **http://127.0.0.1:5000/** (viewer) or
**http://127.0.0.1:5000/video_feed** (MJPEG endpoint). Stop with **Ctrl+C**.
The demo repeats a synthetic image; GPS/AGL are explicitly unavailable.

Stream the real Raspberry Pi camera over your local network:

```bash
python main.py --stream --stream-host 0.0.0.0 --stream-port 5000 --stream-fps 30
```

On the operator PC, open `http://<raspberry-pi-IP>:5000/`. The default bind address
is localhost; `0.0.0.0` makes it reachable over the LAN. This is a small local
monitor without authentication or TLS, intended for a trusted network.

To monitor the simulated mission, including its mock GPS and AGL:

```bash
python main.py --dry-run --stream
```

The server starts in **INIT**, before the camera opens or any simulated movement.
The simulator completes commands instantly, so intermediate states can pass
too quickly to see. After success, the browser remains available until Ctrl+C,
with the final frame labelled **CAMERA: STOPPED**. Without `--stream`, the mission
still exits immediately as before. Failed missions return exit code 1.

Overlay includes mission state, AGL, latitude/longitude, detection status, the
enclosing quadrilateral, circle and center. Preview uses state SCANNING and
does not invent telemetry. Analysis runs independently of capture/encoding;
the latest detection is shown with its source-frame age. Old geometry is hidden
after one second, and missing/stale camera or telemetry data are labelled.
These overlays are for monitoring, not timestamp synchronization for geolocation.

The implementation uses Python's `ThreadingHTTPServer` and Motion JPEG:

- One worker owns camera capture. Analysis reads a fresh frame from a single
  latest-frame buffer, without opening the camera a second time.
- Separate workers encode JPEGs and poll cached telemetry. Blocking analysis or
  `FlightController.execute()` does not stop capture or browser playback.
- Each HTTP client has its own handler; slow clients skip frames and blocked
  writes time out. There is no unbounded frame queue.
- `--stream-fps` limits acquisition/encoding to 30 FPS by default. Actual FPS
  depends on resolution, camera, CPU and network; detection may run more slowly.

`get_telemetry()` must return a thread-safe cached sample promptly even during
motion. A real adapter must update that cache separately from flight commands.
`BufferedCamera` requires bounded reads for mission use; a stuck hardware driver
is reported as stale video/read timeout. The Pi preview backend still has a
blocking device read and has not been promoted to a real flight adapter.

## Complete mission without hardware

```bash
python main.py --dry-run
python main.py --dry-run --mission config/mission.example.json
python main.py --dry-run --image tests/assets/front.png --mission my-mission.json
```

`config/mission.example.json` contains **synthetic** coordinates and FOV values,
not defaults for your actual aircraft/site. An input image must match the
resolution in its mission calibration. The bundled mission example uses
1280×720; static detector fixtures are 900×600.

Without `--image`, dry-run uses the original generated demo image. Flight
commands are recorded and completed instantly by `SimulatedFlightController`;
this tests software integration, not flight dynamics.

Send the result to an explicitly selected PC (only do this when the receiver is ready):

```bash
python main.py --dry-run --udp-host 192.168.1.10 --udp-port 5005
```

Without `--udp-host`, JSON is printed to stdout; logs go to stderr. A PC receiver
must listen on that UDP port. Payload:

```json
{"target_found": true, "coordinates": {"x": 10.3, "y": 13.4}}
```

**x is East and y is North, both in meters relative to configured start point A.**
They are not image pixels or latitude/longitude. `geolocate()` also returns WGS84
latitude/longitude. The sender and receiver must share the configured origin.
UDP success means the datagram was handed to the network, not that the PC
acknowledged receipt. Nothing is transmitted on import or during normal camera preview.

## Modules and state machine

- `navigation.py`: convex quadrilateral validation, area centroid, FOV coverage
  height and high-level TAKEOFF/GOTO/ASCEND command intents.
- `target_detector.py`: preserved detector plus `detect_circle(frame)`, which
  selects the largest qualifying quadrilateral and returns `Point` or `None`.
- `geolocation.py`: image ray, explicit camera/body/world rotations, flat-ground
  intersection and local WGS84 conversion.
- `publisher.py`: console/UDP JSON output.
- `mission.py`: dependency-injected controller and timeout/error handling.
- `simulation.py`: fake flight controller and static-image camera.
- `streaming.py`: MJPEG server, telemetry overlay and bounded latest-frame store.
- `live_camera.py`: background camera capture and detector-to-overlay adapter.

```text
INIT → MOVING_TO_B → CALCULATING_CENTER → ASCENDING → SCANNING
     → TARGET_FOUND → SENDING_DATA → COMPLETE
Any failure → FAILED → adapter.abort() + camera.close()
```

The entire geometric plan is checked before sending any command. The mission
starts on the ground within 2 m of A, takes off to configured transit AGL, moves
to B and then to the area centroid, and climbs to the required scan altitude.
It verifies position/height/orientation in telemetry before scanning.

The area center is the polygon's **area centroid**, not the average of four
corners. For a level, nadir camera the required height is:

`H = margin × max(max_horizontal_offset / tan(horizontal_FOV/2), max_vertical_offset / tan(vertical_FOV/2))`.

Both offsets include the most distant corner relative to the centroid and the
configured yaw. A calibrated principal point away from the image center uses
the narrower camera half-angle for conservative coverage. Heights above the
configured maximum are rejected before takeoff; no descent is planned if the
transit height already covers the area. All corners must be supplied in perimeter order.

Geometric centering **does not guarantee a radio link**, collision-free route,
or terrain clearance. Those require information not present in this repository.

## Coordinates and AGL

- CAMERA: Right / Down / optical Forward.
- BODY FRD: Forward / Right / Down.
- WORLD NED: North / East / Down.

Column-vector rotations use `Rz(yaw) @ Ry(pitch) @ Rx(roll)` with public angles
in degrees. Positive roll lowers the right wing, pitch raises the nose, and yaw
turns North toward East. Yaw must reference true North. Zero camera mount looks
forward; `(0, -90, 0)` explicitly means nadir, image top pointing forward.
Automatic coverage planning currently supports this nadir mount only, with a
level aircraft. Geolocation itself handles other explicit mounts/attitudes.

Ground is `z = altitude_agl_m` in NED. **AGL means height above local ground, not
GNSS ellipsoid/MSL altitude or height relative to takeoff.** Ground-facing rays
are intersected with this plane; upward/horizon rays are rejected.

WGS84 conversion uses meridional/prime-vertical curvature, not a constant
kilometers-per-degree factor. It is a local approximation limited to 1 km,
excluding locations within 1 degree of the poles. The model assumes flat ground,
camera and GNSS at the same origin, and distortion-corrected input frames.
Measured calibration, mount, true attitude, AGL and timestamp-aligned telemetry
are required for real accuracy. RTK alone does not remove these error sources.

## Failure handling and real UAV integration

- No circle within 60 seconds (configurable): FAILED, abort, no target payload.
- Camera/telemetry failure, bad calibration resolution, invalid ray, command
  timeout or UDP send failure: FAILED and the adapter's abort policy is invoked.
- Every exit closes the mission camera; Ctrl+C also aborts the mission.
- `FlightController.execute(command, timeout_s)` must wait for confirmed completion
  and enforce its deadline. `MissionCamera.read(timeout_s)` must be bounded.

No autopilot/protocol is specified, so no live flight adapter has been invented.
Running `--mission ...` without `--dry-run` fails clearly before any hardware
operation. To fly, implement the protocol for your chosen flight controller,
including preflight checks, real AGL, link-loss/abort behavior and a bounded,
time-aligned camera adapter. The existing `PiCamera` remains the preview backend;
its blocking `read()` is not silently used as the timed mission-camera contract.
The controller does not prescribe landing/return after success; define that
policy in the real integration before flight.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python main.py --dry-run
```

The suite preserves the original detector tests. Five static PNG fixtures in
`tests/assets` cover frontal/perspective/rotated views, dim illumination, and
noise plus magenta cast. Their expected centers come from the generating
homographies, with a ±5 pixel assertion per axis. They are synthetic, not field
validation. Regenerate them using `python tests/make_fixtures.py`.
The end-to-end test runs the simulated mission and receives the actual JSON on
a loopback UDP socket at `127.0.0.1`; it requires local socket permission.

Live-stream tests also use loopback HTTP (no camera/UAV). Run the 30 FPS dummy
generator/consumer test with measured throughput output:

```bash
python -m pytest -q -s tests/test_streaming.py
```

It decodes actual multipart JPEGs while another client does not read and a
navigation loop keeps updating. Other tests block analysis and a flight command,
check that fresh frames continue, verify overlays/stale-data handling, simulate
camera loss, and check socket/camera cleanup. Timing assertions allow scheduler
variation; these tests are not a Raspberry Pi performance benchmark.

## Raspberry Pi setup

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-picamera2

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
```

`Picamera2` is intentionally **not** installed with `pip`. It comes from the Raspberry Pi OS
package `python3-picamera2`. The `--system-site-packages` option lets the project virtual
environment see that system package. On macOS/Windows, Picamera2 is not imported unless the
real Raspberry Pi camera backend is actually opened, so `--demo` and `--image` still work.

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
