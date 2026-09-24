# Hackathon

Detektor terče „kolečko uvnitř obdélníku“ pro kameru dronu, webový stream obrazu
a simulace mise. Detekuje se jen tato dvojice: samotné kolečko ani samotný
obdélník nový cíl nezaloží. Detekci ve všech režimech dělá `src/vision.py`.

## Spuštění

Na Raspberry Pi (CSI kamera, výchozí 1280 x 720), obraz v prohlížeči na
`http://<IP Raspberry Pi>:5000/`:

```bash
python main.py --stream --stream-host 0.0.0.0
python main.py --stream --stream-host 0.0.0.0 --ev -2   # tmavší expozice pro bílý list
```

Další režimy:

```bash
python main.py                      # okno s náhledem (přes VNC), Q/Esc konec
python main.py --snapshot test.jpg  # jeden označený snímek, bez okna (SSH)
python main.py --camera 0           # USB kamera / webkamera na PC
python main.py --video let.mp4      # záznam z letu; ve streamu hraje dokola
python main.py --image foto.jpg     # obrázek
python main.py --demo --stream      # vygenerovaná scéna v prohlížeči
python main.py --dry-run            # simulovaná mise (viz níže)
```

Záznam na Pi např. `rpicam-vid -t 30000 --codec libav -o let.mp4`, pak přehrát
přes `--video` (kamera nesmí být zároveň otevřená v `main.py`). V okně klávesy
**1**, **2**, **3** přepínají citlivost hran, **B** vrací výchozí 1.5, **E** přepíná
obraz a hrany používané detektorem. Vyšší citlivost může přidat falešné detekce.
Náhled ukazuje čas zpracování v ms. Růžový nádech kamery se softwarově neodstraňuje.

**Rozlišení:** kolečko 200 mm z výšky 5 m má v 1280 x 720 průměr asi 50 px kolmo dolů
a jen asi 12 x 6 px při náklonu kamery 60°. V 640 x 480 je při 60° pod hranicí
detekce, proto je výchozí 1280 x 720 (`--width`, `--height`).

## Detekce v jednom snímku

Čtyřúhelník: konvexní, uzavřený kontrastní obrys, plocha alespoň 500 px²,
strany alespoň 12 px, pravé úhly se nevyžadují (šikmý pohled). Obrys musí
polygon těsně sledovat, jinak by se za čtyřúhelník vydávala skvrna trávy.
Obrysy se hledají v hranách i v hranách s uzavřenými jednopixelovými mezerami.
Pokud nic nenajdou, záložní průchod spojí dlouhé slabé úsečky přes mezery v rozích.

Kolečko se hledá jen uvnitř masky čtyřúhelníku, s rezervou 2 px od okraje.
Při šikmém pohledu je kolečko elipsa; přijímá se poměr os od 0.35 (~70° od kolmého
pohledu). Obrysy pocházejí z hran i z Otsuova prahování (uzavřené i při slabém
kontrastu). Tvar ověřuje odchylka od elipsy v pixelech, pokrytí obvodu a 4. harmonická
poloměru, která odmítne i malý rozmazaný čtverec. Hough je záloha pro přerušené hrany
při téměř kolmém pohledu. Nerozlišuje význam objektu: kulatý odlesk v obdélníku
může být detekován stejně jako značka.

## Sledování mezi snímky (`src/tracker.py`)

Nový cíl vzniká jen z jediného kandidáta; po 4 navazujících detekcích je
**TERC POTVRZEN**. Při více kandidátech bez sledovaného cíle se nic nepotvrdí.
Potvrzený cíl drží i přes výpadky:

- až 10 snímků bez detekce: poloha se predikuje (oranžově, `measured = False`),
- až 15 snímků jen kolečko bez obdélníku na očekávaném místě (přeexponovaný okraj,
  terč u kraje obrazu); pak je obdélník opět nutný.

Jde o navazování blízkých poloh a velikostí, nikoli záruku identity objektu.
Na statickém obrázku/demu se opakuje tentýž snímek, stabilitu ověřte na videu
nebo živé kameře. Poloha je vyhlazená alfa-beta filtrem.

Sledovaný cíl se hledá jen ve výřezu kolem jeho posledního obdélníku (šedý rámeček
v náhledu); po více než 2 snímcích bez nálezu se znovu prohledává celý snímek.
Na PC to v 1280 x 720 zkrátí detekci z ~18 ms na ~2 ms na snímek. `Observation.offset` dává odchylku cíle
od středu obrazu v rozsahu -1..1 (kladně vpravo a dolů) pro navedení nad střed.

## GPS poloha kolečka (`src/geolocation.py`)

Z pixelu kolečka, úhlů serv a polohy dronu spočítá paprsek a jeho průsečík se zemí:

```python
fix = locate((u, v), DronePose(lat, lon, height, roll, pitch, yaw),
             gimbal_from_servos(x_prikaz, y_prikaz, X_SERVO, Y_SERVO),
             CameraModel.load('camera_calibration.json'), frame_size=(640, 480))
fix.lat, fix.lon, fix.distance, fix.error  # GPS, vodorovná vzdálenost a chyba (1 sigma) v m
```

- Terén je rovný ve výšce místa startu. `height` je ArduPilot `relative_alt` (výška nad
  místem startu z EKF, výchozí zdroj je barometr), přesnější než samotná výška z GPS.
- Z ArduPilotu (MAVLink): `GLOBAL_POSITION_INT` (lat, lon, relative_alt) a `ATTITUDE`
  (roll, pitch, yaw). Serva nejsou stabilizovaná, náklon dronu se proto započítává.
- Závěs: vnější servo osa x (kladně = kamera kouká doprava) nese vnitřní servo osa y
  (kladně = dopředu), 0/0 = kolmo dolů. Při 0/0 míří horní okraj obrazu k přídi
  (jinak `image_top='right' | 'backward' | 'left'`).
- `centering_angles` vrátí úhly serv, které přesunou kolečko do středu obrazu (zámek).
- `combine` spojí opakovaná měření váženě podle chyby: měření nad kolečkem převáží šikmá.

Očekávaná chyba s výchozími nejistotami (serva ±2°, kompas ±3°, náklon ±1°, výška ±1 m,
GPS dronu ±1.5 m), v závorce vodorovná vzdálenost kolečka:

| výška | kolmo | 30° | 45° | 60° |
|---|---|---|---|---|
| 10 m | 1.6 m | 1.8 m (6 m) | 2.1 m (10 m) | 3.0 m (17 m) |
| 20 m | 1.9 m | 2.2 m (12 m) | 2.8 m (20 m) | 4.5 m (35 m) |
| 30 m | 2.2 m | 2.7 m (17 m) | 3.6 m (30 m) | 6.2 m (52 m) |

Šikmo tvoří největší část chyby serva, kolmo GPS dronu. Proto hrubý odhad ze středu
prostředí a zpřesnění přeletem nad kolečko.

Kalibrace před letem:

1. **Kamera:** vytiskněte šachovnici 10 x 7 polí (9 x 6 vnitřních rohů), změřte stranu pole
   a spusťte `python calibrate_camera.py --picamera 0 --square 25`. Vznikne
   `camera_calibration.json` (RMS by mělo být pod 0.5 px). Bez kalibrace se použije jmenovité
   zorné pole 53.5° x 41.4° (OV5647 + 3.6 mm) bez korekce zkreslení.
2. **Serva MG996R:** skutečný úhel neodpovídá příkazu přesně. Pro každé servo nastavte
   příkazy -60, -30, 0, 30, 60°, změřte skutečný sklon kamery (sklonoměr v telefonu na
   kameře) a proložte přímku: `ServoAxis(sign, scale, offset)`. Znaménko: kladný úhel x
   musí naklonit kameru doprava, kladný úhel y dopředu.
3. **Natočení obrazu:** při 0/0 dejte před příď předmět; musí být u horního okraje obrazu.
4. **Ustálení:** servo se otáčí asi 0.2 s na 60° a úhel nehlásí. Pro měření polohy
   berte snímky až po ustálení (alespoň 0.3 s po posledním pohybu) a v klidném visu.

## Webový stream (live video in a browser)

No GUI window or additional web framework is needed. Test on a generated scene:

```bash
python main.py --demo --stream
```

Any source works with `--stream` (`--camera`, `--video`, `--image`, CSI camera).

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

## Mise bez hardwaru (complete mission without hardware)

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

## Moduly a stavy mise

- `navigation.py`: convex quadrilateral validation, area centroid, FOV coverage
  height and high-level TAKEOFF/GOTO/ASCEND command intents.
- `vision.py`: detector used by the mission and stream. `detect_circle(frame)`
  returns `Point` only for a tracker-confirmed target measured in that frame.
- `target_detector.py`: the original detector of the test branch, kept for its
  tests and comparison; it misses small (200 mm at 5 m) targets.
- `geolocation.py`: `geolocate()` for the fixed mount used by the mission and
  `locate()` for the servo gimbal (see above), both with local WGS84 conversion.
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

## Souřadnice mise a AGL

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

## Chyby a napojení na skutečný dron

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

## Instalace na Raspberry Pi

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

## Testy

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/synthetic_scene.py` vykresluje terč na trávě perspektivní kamerou (náklon,
rotace, výška, šum, rozmazání, expozice, JPEG) i s návnadami: samotné kolečko,
prázdný list, čtverec v listu, samotný prstenec; `small_sheet` a `small_outline`
mají skutečnou velikost (kolečko 200 mm). Syntetická scéna neověřuje reálnou
kameru: laďte na záznamech z letu přes `--video`.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python main.py --dry-run
```

The suite preserves the original detector tests; Vision is tested in
`tests/test_web_integration.py`, `tests/test_perspective_tracking.py` and others. Five static PNG fixtures in
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
