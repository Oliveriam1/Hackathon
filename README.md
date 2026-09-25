# Míření na červenou tečku

Program na Raspberry Pi najde kamerou červenou tečku a servy natočí kameru
přesně na její střed. Výsledkem je **úhel kamery na střed tečky**. Když zná
polohu dronu, podprogram z úhlu spočítá **GPS souřadnice tečky**.

Navigaci dronu (let na pozici, GPS, výška) řeší jiná část týmu. Tento program
nekomunikuje s letovým kontrolérem.

```
kamera ──► detekce tečky ──► serva míří na střed ──► ÚHEL (right, forward)
                                                            │
                        poloha dronu (GPS, výška, kurz) ────┴──► SOUŘADNICE tečky
```

## Spuštění

```bash
# jen úhel na tečku (na dronu, přes SSH)
python3 main.py --picamera 0 --headless

# úhel + souřadnice, dron stojí na 50.0875123, 14.4213456, 5 m nad zemí, příď na sever
python3 main.py --picamera 0 --headless --drone-pose 50.0875123 14.4213456 5.0 0 --result tecka.json --once

# polohu dronu dodá jiný program do souboru (čte se při každém výsledku)
python3 main.py --picamera 0 --headless --pose-file pose.json --result tecka.json
```

`pose.json`:
```json
{"latitude_deg": 50.0875123, "longitude_deg": 14.4213456, "height_m": 5.0, "heading_deg": 0.0}
```
(volitelně `roll_deg`, `pitch_deg`, když dron není vodorovně).

Výstup v terminálu:
```
CENTERED   kamera R= +12.40 F= -3.10 | detekce   38 ms, 12.1 fps | odchylka  +1.2, -0.8 px | úhel na tečku R= +12.43 F= -3.08
ÚHEL NA STŘED TEČKY: right +12.43°, forward -3.08° (medián 20 snímků, rozptyl 0.05°)
SOUŘADNICE TEČKY: 50.08750991, 14.42135405 (1.10 m od dronu, azimut 103°, odhad chyby ±18 cm)
```

## Úhly

| úhel | význam |
| --- | --- |
| `right` | náklon kamery doprava (+) / doleva (−) vůči dronu, stupně |
| `forward` | náklon kamery dopředu (+) / dozadu (−) vůči dronu, stupně |
| 0 / 0 | kamera míří kolmo dolů |

Úhel na tečku = skutečné natočení kamery + zbývající odchylka tečky od středu
obrazu přepočtená přes model objektivu. Je tedy přesný i tehdy, když serva stojí
o kousek vedle. Výsledek je medián z `--samples` (20) snímků, kdy byla tečka
stabilně ve středu obrazu.

Stavy: `SEARCHING` (tečka není vidět), `SCANNING` (serva prohledávají okolí),
`TRACKING` (dotahuje tečku do středu), `CENTERED` (tečka ve středu ≥ 0,5 s,
úhel platný), `LOST`, `AMBIGUOUS` (víc podobných teček – neměří se).

## Použití z vlastního programu

```python
from src.locate import DronePosition, locate_target

drone = DronePosition(50.0875123, 14.4213456, height_m=5.0, heading_deg=0.0)
target = locate_target(drone, right_deg=12.43, forward_deg=-3.08)
print(target.latitude_deg, target.longitude_deg, target.error_m)
```

Míření po snímcích: `src.aim.Aimer` (jeden krok na snímek) a `src.aim.AngleAverager`
(medián úhlu), viz `src/app.py`.

## Přesnost

Trojúhelník je nejpřesnější, když kamera míří **skoro kolmo dolů**. Z výšky 5 m:

| zdroj chyby | kolmo dolů | 45° šikmo |
| --- | --- | --- |
| úhel kamery 1° | 9 cm | 17 cm |
| náklon dronu 1° | 9 cm | 17 cm |
| výška o 10 cm | 0 cm | 10 cm |
| kurz dronu 2° | 0 cm | 17 cm |

Proto: **kalibrovat serva** i kameru a nechat dron zastavit co nejblíž nad
tečkou. Přesnost polohy dronu (GPS) se k tomu přičítá celá – tu dodává navigace.

## Nastavení (jednou)

**Raspberry Pi:**
```bash
sudo apt install python3-opencv python3-picamera2 pigpio python3-pigpio
sudo systemctl enable --now pigpiod
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
```

**Serva** (2× MG996R, BCM 18 = vnější/doprava, BCM 13 = vnitřní/dopředu):
1. `python3 main.py --picamera 0` (s oknem) – posuňte tečku doprava v obraze;
   kamera se musí natočit za ní. Když jde opačně: `--servo-x-dir -1` / `--servo-y-dir -1`.
2. Když tečka posunutá k přídi dronu není v obraze nahoře: `--image-top right|backward|left`.
3. Kalibrace úhlů (mobil se sklonoměrem na kameře):
   `python3 tools/calibrate_servos.py --output servo_calibration.json`,
   pak `--servo-calibration servo_calibration.json`.

**Kamera** (šachovnice 9×6, pole 25 mm, stejné rozlišení jako main.py):
```bash
python3 calibrate_camera.py --picamera 0 --output camera_calibration.json
python3 main.py --picamera 0 --headless --camera-calibration camera_calibration.json
```

**Detekce na skutečném světle:** `python3 main.py --picamera 0 --tune` (posuvníky prahů;
program vypíše parametry k použití), bez okna `--diagnostics logs/` uloží snímky.

## Ověření přesnosti na zemi

1. Dron na stojan, kamera 2–5 m nad zemí, změřte výšku kamery.
2. Tečku položte do změřené vzdálenosti (např. 1,5 m před a 1 m vpravo od bodu pod kamerou).
3. `python3 main.py --picamera 0 --headless --drone-pose <lat> <lon> <výška> <kurz> --once`
   – porovnejte `north_m`/`east_m` ve výsledku (`--result`) se změřenými hodnotami.

## Další parametry

`python3 main.py --help` – zdroj (`--video` záznam, `--image` fotka), pevná kamera
bez serv (`--fixed-camera 0 0`), výpočet bez hýbání servy (`--dry-run`), rychlost
serv, vypnutí prohledávání (`--no-scan`), prahy detekce, JSON stavu každého snímku
(`--output`), `--snapshot foto.jpg`.

## Struktura

| soubor | co dělá |
| --- | --- |
| `main.py` | spuštění |
| `src/app.py` | hlavní smyčka |
| `src/aim.py` | **míření: úhel na střed tečky** |
| `src/locate.py` | **podprogram: souřadnice z polohy dronu a úhlu** |
| `src/vision.py`, `red_detector.py`, `tracker.py`, `circle_detector.py`, `target_lock.py` | detekce a sledování tečky |
| `src/gimbal.py`, `gimbal_controller.py` | serva, regulátor míření, prohledávání |
| `src/geometry.py` | model kamery a úhly |
| `src/camera.py`, `pi_camera.py`, `sources.py` | kamera |
| `src/config.py`, `preview.py`, `diagnostics.py` | parametry, okno, ladicí snímky |
| `calibrate_camera.py`, `tools/calibrate_servos.py` | kalibrace |
| `tests/` | `python3 -m unittest discover -s tests` |
