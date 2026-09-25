# Skutečný let – `tools/fly_mission.py`

Spojuje kameru (tečka + páska), řadič mise a ArduPilot přes MAVLink.
Průběh: kontrola na zemi → GUIDED → arm → vzlet na `--takeoff-height` →
hledání po řádcích → přílet nad tečku → centrování kamery → držení.

```bash
# 1) ArduPilot SITL bez kamery (vzlet, trasa, hranice, ACK, převzetí)
python3 tools/fly_mission.py --connect tcp:127.0.0.1:5760 --field-bounds -10 10 -10 10 --fake-camera --yes

# 2) Skutečný dron
python3 tools/fly_mission.py --connect /dev/serial0 --baud 921600 --real-flight \
    --field-bounds -10 10 -10 10 --takeoff-height 5 --output mise.jsonl \
    --camera-args "--camera-calibration kalibrace.json"
```

Start (0, 0) je místo, kde dron stojí při spuštění programu. `--field-bounds`
jsou metry sever/východ od něj. Program vypíše plán a čeká na napsání `START`.

## Jak mise funguje

| Fáze | Co se děje |
| --- | --- |
| Start | Zachytí `LOCAL_POSITION_NED` + GPS jako počátek. Stroj musí být nearmovaný na zemi. |
| Vzlet | `MAV_CMD_DO_SET_MODE` GUIDED → `ARM` → `NAV_TAKEOFF`. Každý krok čeká na `COMMAND_ACK` **a** telemetrii. |
| Hledání | Řádky po 3 m uvnitř pole, kamera kolmo dolů (`SEARCH`). Rychlost 20 Hz `SET_POSITION_TARGET_LOCAL_NED`, výška se drží P regulátorem. |
| Páska | Úseky pásky se promítnou na zem a po 2 snímcích se stanou **zakázanou čarou** (nekonečná přímka, povolená strana = strana startu). Body trasy se ořežou 1 m před ní, cíl za páskou se ignoruje. Pokud páska vyjde najevo až za dronem, vrací se kolmo zpět (`TAPE_RETREAT`). |
| Tečka | Poloha = poloha dronu při snímku + posun vypočtený z kamery. Chyba GPS se tím vyruší. Závěs přepne na `TRACK` a centruje tečku. |
| Konec | Dron nad tečkou (≤ 0.4 m, 1 s klidu) **a** kamera `IMAGE_CENTERED` 0.5 s → `HOLDING_TARGET`, souřadnice tečky ve výpisu a logu. |

Rychlost je omezená tak, aby dron stihl zastavit před páskou, kterou uvidí až
na okraji záběru: 5 m → 1.24 m/s, 4 m → 1.0 m/s, 8 m → 1.8 m/s (a nejvýš
`--max-speed`, výchozí 1 m/s). Pod 4 m výšky kamera vidí pásku příliš pozdě.

## Přesné měření červené vůči zelené (cíl: chyba pod 25 cm)

Dron vždy startuje na zelené tečce se známou souřadnicí. GPS dronu během letu
ujíždí (typicky 0.3–2 m za pár minut), a proto se souřadnice červené počítá
z rozdílu dvou kamerových měření:

`červená = zadaná zelená + (červená změřená − zelená změřená)`

Průběh po nalezení červené (`--precision-loops 2`, výchozí):

1. `MEASURE_GREEN_START` – hned po vzletu visí nad zelenou, 15 měření.
2. `MEASURE_RED` – visí nad červenou, kamera kolmo dolů, 15 měření (medián).
3. `RETURN_GREEN` → `MEASURE_GREEN` – zpět nad zelenou, znovu změřit.
4. `RETURN_RED` → `MEASURE_RED` → `RETURN_GREEN` → `MEASURE_GREEN` – druhé kolo.
5. `DONE` – poloha zelené v čase měření červené se interpoluje ze sousedních
   měření zelené, takže se lineární část driftu GPS odečte. Výsledek: WGS84
   souřadnice, vzdálenost a azimut od zelené.

Když bod po návratu není v záběru (drift), dron obhlédne okolí po čtverci.
Po celý let drží stejný kurz (yaw ze startu), aby byl záběr kamery vždy stejně
natočený a chyba montáže kamery se v rozdílu odečetla. `--finish land` na konci
přistane na zelené.

```bash
python3 tools/fly_mission.py --connect /dev/serial0 --baud 921600 --real-flight \
    --field-bounds -10 10 -10 10 --green 50.0875123 14.4213456 --green-diameter-m 0.2
python3 tools/run_mission.py --gps-drift 0.5 --camera-noise 0.1     # simulace s driftem
```

**Bez `--green` se jako zelená použije GPS startu** – vzdálenost od zelené je
pak přesná, ale absolutní souřadnice nesou chybu GPS (metry). Souřadnici zelené
tedy vždy zadejte.

Simulace (12 běhů, drift GPS 0.5 m, šum kamery 10 cm): jen z GPS medián 58 cm,
pod 25 cm 1 z 11; se zelenou medián 5.5 cm, max 16 cm, pod 25 cm 11 z 11.
Simulace ale předpokládá nezkreslený šum kamery – skutečnou přesnost ověřte
na zemi: položte červenou tečku do známé vzdálenosti od zelené a porovnejte.

Detektor zelené (`marker_detector.py`) hledá sytě zelený kulatý izolovaný bod
o velikosti odpovídající `--green-diameter-m` a výšce; tráva by neměla projít.
Prahy HSV je nutné ověřit na skutečném poli (diagnostické snímky).

## Převzetí a poruchy

- **Přepnutí režimu na vysílačce** (LOITER/ALT_HOLD/…) = okamžité převzetí.
  Program přestane posílat povely a skončí.
- **Ctrl+C**: nulová rychlost 1 s, pak přepnutí do `--exit-mode` (výchozí LOITER).
- Stará telemetrie, ztráta kamery, odmítnutý povel, timeout, ztráta výšky →
  `FAILSAFE` = nulová rychlost ve vzduchu. Pilot pak přebírá.
- Spadne-li Pi, ArduPilot po 3 s bez povelu sám zastaví a drží polohu.

## Zapojení a nastavení ArduPilotu

1. **UART**: Pi GPIO14 (TX) → RX portu TELEM2, GPIO15 (RX) → TX, společná zem.
   Na Pi: `raspi-config` → Serial: login shell **ne**, hardware **ano** → `/dev/serial0`.
2. **Parametry** (Mission Planner / QGC):
   - `SERIAL2_PROTOCOL = 2` (MAVLink2), `SERIAL2_BAUD = 921` → `--baud 921600`
   - `FENCE_ENABLE = 1`, `FENCE_TYPE = 7` (výška + kruh + polygon), `FENCE_ALT_MAX = 15`,
     `FENCE_ACTION = 1` (RTL) nebo 4 (Brake). **Polygon nakreslete podle skutečné pásky** –
     je to záloha nezávislá na Pi a kameře.
   - `FS_THR_ENABLE`, `BATT_FS_LOW_ACT`, `FS_GCS_ENABLE = 0` (Pi není GCS; při výpadku
     Pi zastaví guided timeout).
   - Přepínač režimu na vysílačce s pozicemi GUIDED / LOITER / LAND.
3. **Kamera a závěs**: ověřte `--servo-x-dir`/`--servo-y-dir` a `--image-top` podle
   DETECTION.md, kalibrujte kameru (`calibrate_camera.py`) a předejte ji přes `--camera-args`.
   Při `--gimbal fixed` musí kamera mířit pevně kolmo dolů.
4. `pip install pymavlink` na Pi (je v `requirements-pi.txt`).

## Postup testování

1. `python3 -m unittest discover -s tests -q` – vč. modelu autopilota (`test_flight_runner.py`).
2. `python3 tools/run_mission.py --tape 5 -12 5 12` – simulace s páskou.
3. ArduPilot SITL (`sim_vehicle.py -v ArduCopter`) + `--fake-camera`.
4. Na zemi bez vrtulí: `--connect /dev/serial0 ... --real-flight`, zkontrolovat, že
   se zachytí start, ukazuje se poloha, páska a tečka se hlásí v logu. Ukončit před START.
5. Venku s vrtulemi, malé pole, nízká výška 4–5 m, pilot s prstem na přepínači.

## Známá omezení

- Pole je obdélník v osách sever/východ. Šikmé pole: zvolte obdélník uvnitř
  a spolehněte se na pásku + fence ArduPilotu.
- Páska se promítá z úhlů serv bez zpětné vazby a z nesynchronizovaného času
  snímku; chyba roste s náklonem kamery. Proto se hledá s kamerou kolmo dolů.
- Detekce pásky i tečky byla laděna na syntetických obrazech; na skutečném poli
  je nutné ověřit prahy (`--diagnostics`).
