# UAV Vision: tři samostatné programy

Python 3.11+. Žádný program neposílá příkazy motorům, waypointy ani povely k letu.
Každý program má vlastní CLI, logging na **stderr** a JSON výsledky na **stdout**.
Import modulů neotevírá hardware.

## 1. Vision — kamera a detekce

```sh
python vision_main.py --camera-backend opencv --camera
python vision_main.py --camera-backend picamera2
python vision_main.py --debug --camera
```

Vision načítá snímky, hledá GREEN referenci a RED kandidáty a publikuje
`VisionResult` jako jeden JSON objekt na řádek. Neprovádí navigaci ani GNSS výpočet.
Bez `--camera`/`--preview` běží bez okna, vhodně pro Raspberry Pi přes SSH.
`--debug` pouze zapíná podrobnější logování. `--max-frames 10` omezí počet snímků.
U náhledu ukončí aplikaci `q`, Escape nebo zavření okna; bez okna Ctrl+C.

Konfigurace rozlišení, backendu, minimální plochy kontury a tolerance centrování
je v `config/vision.json`; lze dodat jiný soubor přes `--config`. CLI backend/index
přepisují příslušná nastavení ze souboru. Skutečné rozlišení se vypíše po prvním
snímku a je součástí každého výsledku. HSV konstanty jsou v `src/detector.py`.

Zelená poskytuje střed, pixelovou i normalizovanou odchylku a `centered`.
Červená poskytuje **kandidáty**, nikoli potvrzené cíle: střed, normalizovaný střed,
plochu a bounding box `(x, y, width, height)`. Červeno-bílá páska může být také
kandidátem. `VisionProcessor(candidate_filter=...)` umožňuje pozdější filtr podle
kruhovitosti, poměru stran, solidity, hranic nebo historie; nic se zatím automaticky
nepotvrzuje. Detektor zůstává HSV + morphology + plocha kontury.

Výstup například:

```json
{"timestamp":123.5,"frame_width":320,"frame_height":240,"green_detection":null,"centering":{"detected":false,"center":null,"error_x":null,"error_y":null,"normalized_error_x":null,"normalized_error_y":null,"centered":false},"red_candidates":[]}
```

`timestamp` je `time.monotonic()` při převzetí snímku, v sekundách. Není to UTC
ani přesný čas expozice. Platí pro místní synchronizaci ve stejném běhu OS;
po restartu nebo mezi různými stroji tyto hodnoty nelze přímo porovnávat.

Testovací růžová zůstala zachována:

```sh
python vision_main.py --camera --test-pink
```

Je vypnutá ve výchozím stavu, zobrazuje `PINK TEST` a nevstupuje do datového
kontraktu GREEN/RED. Jde o detekci v každém snímku, bez trvalých ID objektů.
Pro odstranění stačí odstranit `detect_pink_test`, konstanty `*_PINK_TEST`,
přepínač a růžové bloky náhledu. Ostatní detekce se nemění.

## 2. Navigation — stav centrování

```sh
python navigation_main.py --dry-run
python navigation_main.py --input vision.jsonl
python vision_main.py --camera-backend opencv | python navigation_main.py --input -
```

Navigation přijímá `CenteringResult` nebo `centering` z celého Vision JSON.
Neimportuje OpenCV/NumPy, nezná HSV a nic nedetekuje.

- Bez vstupu nebo při neplatném vstupu: `IDLE`.
- Zelená není vidět: `SEARCHING_REFERENCE`.
- Zelená je vidět mimo toleranci: `CENTERING`.
- Zelená je uvnitř tolerance: `CENTERED`.

Výstup je `NavigationDecision`: stav, `horizontal_error`, `vertical_error`.
Chyby jsou normalizované obrazové odchylky, **ne rychlosti nebo pohybové příkazy**.
Kladné X je doprava v obraze, kladné Y dolů. `--dry-run` použije označený
syntetický vstup s chybami −0.13 a +0.05. Skutečný flight-controller adapter
ani pohybové rozhraní zatím nejsou implementované.

## 3. Geolocation — pixel na geografické souřadnice

```sh
python geolocation_main.py --dry-run
python geolocation_main.py --pixel 832 441 \
  --calibration config/camera_calibration.json --telemetry telemetry.json
```

Geolocation dostane pixel a explicitní telemetrii/kalibraci. Neotevírá kameru,
nedetekuje barvy a nespoléhá na Navigation nebo grid. `--dry-run` je jediný režim,
kde CLI vytváří testovací kalibraci a telemetrii: střed snímku, kamera dolů,
AGL 10 m, orientace 0/0/0 a pozice 50° N, 14° E. Výsledek má `synthetic: true`,
North/East přibližně nula a geografickou pozici přibližně shodnou s UAV.

`config/camera_calibration.json` je záměrně **nevyplněná šablona**. Doplňte:

- `intrinsics`: fx, fy, cx, cy v pixelech, width a height;
- volitelně standardní OpenCV `distortion_coefficients` (4/5/8/12/14 hodnot);
- `mount`: roll_deg, pitch_deg, yaw_deg podle skutečné montáže.

Bez platné kalibrace se vypíše `CAMERA CALIBRATION REQUIRED` a program skončí
kódem 1. Bez telemetrie vypíše `TELEMETRY REQUIRED`. FOV se nikdy neodhaduje;
`CameraIntrinsics.from_fov(...)` je pouze utilita pro **známé, explicitně zadané** FOV.
Rozlišení, výřez, rotace i zrcadlení vstupního obrazu musí odpovídat kalibraci.

Telemetrie JSON má tvar (následující hodnoty jsou **pouze příklad**, nikoli default):

```json
{
  "position": {"latitude": 50.0, "longitude": 14.0},
  "altitude_agl_m": 10.0,
  "attitude": {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
  "timestamp": null
}
```

AGL znamená výšku nad místní zemí. **GNSS výška nad elipsoidem, výška nad mořem
ani výška relativně ke startu nejsou automaticky AGL.** Snapshot musí odpovídat
expozici snímku. Volitelný timestamp telemetrie má stejný monotonic základ jako
Vision; tato verze zatím neprovádí automatické párování ani kontrolu stáří.
`TelemetryProvider` je nezávislý na protokolu a `StaticTelemetryProvider` slouží
pro explicitně zadané testovací/replay vzorky.

Výpočet:

```text
pixel → odstranění distortion → jednotkový paprsek K⁻¹
      → camera-to-body → body-to-NED → průsečík se zemí
      → North/East offset → lokální WGS84 → latitude/longitude
```

Soustavy:

- CAMERA: X doprava, Y dolů, Z opticky dopředu.
- BODY FRD: X dopředu, Y doprava, Z dolů.
- WORLD NED: X sever, Y východ, Z dolů.

Používáme sloupcové vektory a aktivní pravotočivé rotace `Rz(yaw) @ Ry(pitch) @ Rx(roll)`.
API používá stupně. Kladný roll sklápí pravé křídlo, pitch zvedá příď a yaw otáčí
ze severu na východ. Yaw musí být vůči skutečnému severu.

Nulový mount znamená kameru dopředu: Right/Down/Forward kamery mapuje na
Right/Down/Forward těla. Mount rotace se aplikuje na tuto základnu v FRD.
Výslovně zvolený `CameraMount(0, -90, 0)` míří dolů, horní část obrazu dopředu.
Při nulové orientaci UAV pak pixel vpravo znamená východ a pixel nahoře sever.
Montáž není nikde předpokládaná jako produkční default.

Země je rovina `z = altitude_agl_m` v NED. Paprsek nad horizontem nebo téměř
vodorovný vyvolá chybu. WGS84 používá meridiánový a příčný poloměr křivosti;
není použita konstanta kilometrů na stupeň. Aproximace je pro krátké vzdálenosti,
odmítá offsety nad 1 km a nenulové offsety do 1° od pólů.

## Předávání dat a původní main.py

`ConsolePublisher` vydává JSON Lines; protokol `Publisher` dovoluje později doplnit
jiný lokální výstup. Žádný broker, síťové API nebo cloud není potřeba.
Vision → Navigation lze propojit přímo přes stdout, jak ukazuje příkaz výše.
Geolocation CLI zpracovává jeden pixel a jeden telemetrický snapshot. Vyšší vrstva
může pro každý kandidát samostatně zavolat:

```python
from src.geolocation import TargetGeolocator

locator = TargetGeolocator(intrinsics, camera_mount)
for candidate in vision_result.red_candidates:
    location = locator.geolocate(candidate.center, telemetry)
```

Výsledek polohy **nepotvrzuje**, že kandidát je správný soutěžní objekt.

`main.py` zůstává kompatibilní launcher původního náhledu (`--camera`, `--test-pink`,
`--dry-run`). `src.camera.run_camera_preview` je tenký wrapper nové Vision vrstvy.
Pro JSON výstup a Pi backend používejte `vision_main.py`. `field.py` je v aktuálním
checkoutu pouze místo pro budoucí volitelný grid/debug; geolokace jej neimportuje.

## macOS setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python vision_main.py --camera-backend opencv --camera
```

Python musí být alespoň 3.11. Aplikace vyžaduje pouze NumPy a OpenCV.

## Raspberry Pi setup

Na Raspberry Pi OS s Pythonem 3.11+:

```sh
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-opencv python3-picamera2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements-pi.txt
python vision_main.py --camera-backend picamera2
```

`requirements-pi.txt` nyní nepotřebuje žádné pip balíčky; NumPy, OpenCV a Picamera2
pocházejí ze systému. Tím se nepřepisují systémové knihovny další kopií OpenCV.
Picamera2 se importuje až při `start()`, takže import i testy fungují na Macu.
Backend používá `create_preview_configuration`, `capture_array("main")` a
ukončuje kameru přes `stop()`/`close()`. Formát Picamera2 `RGB888` poskytuje BGR
pole pro OpenCV; viz [oficiální Picamera2 manuál](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).
Pro okno náhledu přidejte `--camera`; potřebuje grafické prostředí.

## Testy bez hardwaru

```sh
python -m unittest discover -v
python vision_main.py --help
python navigation_main.py --help
python geolocation_main.py --help
python vision_main.py --dry-run
python navigation_main.py --dry-run
python geolocation_main.py --dry-run
```

Testy používají syntetické snímky a mock kamer. Ověřují i oddělení importů,
přenos JSON a nepřítomnost potvrzení cíle u červeno-bílé pásky.

## Current limitations

- Skutečný flight-controller adapter není implementovaný. Žádné flight commands.
- Je potřeba skutečný zdroj AGL a platná GNSS/attitude telemetrie.
- Kalibrace, montáž a časové sladění snímků s telemetrií musí odpovídat realitě.
- Červeno-bílá hranice potřebuje další filtrování; nyní vracíme pouze kandidáty.
- Předpokládáme rovnou zem a společný počátek kamery, těla a GNSS antény.
- RTK samo nezaručuje přesnost cíle: chyby AGL, roll/pitch/yaw, kalibrace,
  montáže, distortion, středu detekce a tvaru terénu se promítají do výsledku.
- Pi backend je otestovaný s mock API; skutečná Pi kamera vyžaduje ověření na zařízení.
