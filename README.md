# Hackathon – míření na červenou tečku

Kamera na Raspberry Pi najde červenou tečku (detekce 1:1 podle `red_tracker.py`),
servy natočí kameru na její střed a vypočítá **úhel na střed tečky**. Se známou
polohou dronu z úhlu spočítá **GPS souřadnice tečky**. Stav i výsledek může
posílat jako **telemetrii přes UDP** (WiFi) na notebook.

```
kamera ─► detekce tečky ─► serva míří na střed ─► ÚHEL (right, forward)
                                                        │
                   poloha dronu (GPS, výška, kurz) ─────┴─► SOUŘADNICE ─► telemetrie UDP
```

## Spuštění na dronu

```bash
# 1) jen detekce (jako dřív) – POZOR: --altitude = skutečná výška kamery nad zemí
python3 main.py --jako-red-tracker --headless --status --altitude 5

# 2) + serva míří na střed tečky, vypisuje úhel
python3 main.py --jako-red-tracker --headless --status --aim --altitude 5

# 3) + souřadnice tečky a telemetrie na notebook
python3 main.py --jako-red-tracker --headless --status --aim \
    --drone-pose 50.0875123 14.4213456 5.0 0 --send 192.168.1.20:5005 --result tecka.json
```

Na notebooku (IP zjistíte `ipconfig`):
```bash
python3 tools/receive_telemetry.py --port 5005 --log let.jsonl
```

`--drone-pose LAT LON VÝŠKA KURZ` – GPS dronu, výška kamery nad zemí (m), kurz přídě
od severu po směru hodinových ručiček. Když polohu dodává jiný program, zapisuje ji do
souboru a použije se `--pose-file pose.json`:
```json
{"latitude_deg": 50.0875123, "longitude_deg": 14.4213456, "height_m": 5.0, "heading_deg": 0.0}
```
S `--drone-pose` se jeho výška použije i jako `--altitude`.

## Výstup

```
ÚHEL NA STŘED TEČKY: right +12.43°, forward -3.08° (medián 20 snímků, rozptyl 0.05°)
SOUŘADNICE TEČKY: 50.08750991, 14.42135405 (1.13 m od dronu, azimut 104°, odhad chyby ±18 cm)
```
- `right` = náklon kamery doprava (+) / doleva (−), `forward` = dopředu (+) / dozadu (−),
  0/0 = kolmo dolů. Úhel zahrnuje i zbývající odchylku tečky od středu obrazu,
  takže je přesný, i když serva stojí o kousek vedle.
- Stavy míření: `SEARCHING`, `SCANNING` (serva prohledávají okolí), `TRACKING`,
  `CENTERED` (tečka ve středu ≥ 0,5 s – teprve pak se úhel započítá), `LOST`.
- Bez `--aim` serva stojí a úhel se počítá z polohy tečky v obraze.

Telemetrie (UDP, jeden JSON na paket): `{"type": "aim", ...}` stav míření max. 10× za s
(`--send-rate`), `{"type": "result", ...}` výsledek (úhel, poloha dronu, souřadnice tečky).

## Z vlastního programu

```python
from src.locate import DronePosition, locate_target
target = locate_target(DronePosition(50.0875123, 14.4213456, height_m=5.0, heading_deg=0.0),
                       right_deg=12.43, forward_deg=-3.08)
print(target.latitude_deg, target.longitude_deg, target.error_m)
```

## Nastavení (jednou)

1. **Směr serv:** `python3 main.py --jako-red-tracker --aim --altitude 1` s oknem, tečku na
   papíře posouvejte – kamera ji musí následovat. Jinak `--servo-x-dir -1` / `--servo-y-dir -1`.
   Když posun tečky k přídi dronu není v obraze nahoru: `--image-top right|backward|left`.
2. **Kalibrace serv** (mobil se sklonoměrem přiložený na kameru):
   `python3 tools/calibrate_servos.py --output servo_calibration.json`,
   pak `--servo-calibration servo_calibration.json`. 1° chyby = 9 cm na zemi z 5 m.
3. **Kalibrace kamery** (šachovnice): `python3 calibrate_camera.py --picamera 0`
   ve stejném rozlišení jako let (1296×972), pak `--camera-calibration camera_calibration.json`.

## Přesnost

Nejpřesnější je, když kamera míří skoro **kolmo dolů**. Z 5 m: 1° chyby úhlu nebo náklonu
dronu = 9 cm kolmo dolů, 17 cm při 45°; chyba výšky 10 cm = 0 cm kolmo dolů, 10 cm při 45°.
Chyba GPS polohy dronu se přičítá celá (tu dodává navigace).

**Ověření na zemi:** kamera ve známé výšce, tečka ve změřené vzdálenosti, spusťte
s `--drone-pose` a porovnejte `north_m` / `east_m` ve výsledku se změřenými hodnotami.

## Testy

```bash
python3 -m unittest discover -s tests
```
