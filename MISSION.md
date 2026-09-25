# Řízení celé mise

Společný `MissionController` propojuje kontrolu startu, potvrzení režimu a
armování, vzlet, hledání, navádění, ustálení, potvrzení kamery a výstup souřadnic.
Aktuálně má spustitelný **kinematický simulační backend**. `main.py` dál slouží
kamerové pipeline, lokalizaci a volitelnému ovládání serv. Řadič v něm zůstává
vypnutý a žádné nové letové povely neposílá.

## Spuštění celé mise

```bash
python3 tools/run_mission.py --seconds 180
python3 tools/run_mission.py --height 5 --start 50 14 --seconds 300 --output mission.jsonl
python3 tools/run_mission.py --no-target
python3 tools/run_mission.py --moving
python3 tools/run_mission.py --target-loss 125 129
python3 tools/run_mission.py --fault map --fault-at 20 --seconds 30
python3 tools/run_mission.py --fault reject_arm --seconds 2
```

Čas běží výpočetně, ne v reálném čase. Výchozí souřadnice 50°, 14° jsou fiktivní.
Výchozí pole je syntetický obdélník ±10 m, tečka je 8 m severně a 5 m východně
od startu a ideální senzor ji vidí do 3 m. Backend simuluje potvrzení povelů,
telemetrii i zaměření kamery po 0.5 s. Nespouští skutečnou obrazovou detekci,
servo ani MAVLink. Testuje návaznost modulů a reakce na jejich vstupy; neověřuje
přesnost obrazové detekce, dynamiku skutečného stroje nebo ArduPilot.

## Průchod

1. `PREFLIGHT`: čerstvá telemetrie, kamera, ověřený obvod shodný s trasou,
   nearmovaný stroj na zemi do 0.5 m od startovní reference.
2. `SETTING_GUIDED` a `ARMING`: jednorázové požadavky s `request_id`.
   Ke změně stavu je nutné odpovídající kladné potvrzení **i** telemetrie.
   Požadavek se při chybě automaticky neopakuje.
3. `TAKING_OFF`, `TAKEOFF_SETTLING`: potvrzený vzlet, dosažená výška a souvislé
   ustálení. Vodorovné povely jsou do té doby nulové.
4. `SEARCHING`: řádky uvnitř zóny. Rychlost se omezuje podle odstupu od hranice,
   brzdné dráhy a reakční doby. Každý povel prochází `ZoneGuard`.
5. `ACQUIRING_TARGET`, `APPROACHING`, `CENTERING`: čerstvý, platný cíl v lokální
   soustavě. Cíl mimo zónu nebo s příliš velkou nejistotou není přijat.
6. `CENTERING_CAMERA` / `HOLDING_TARGET`: fyzický přílet podle polohy a rychlosti
   nestačí; pro výsledek se vyžaduje i potvrzení zaměření kamery. Při pohybu cíle
   se navádění znovu aktivuje. Výchozím závěrem je držení, nikoli přistání.

V JSONL je `command` (záměr), `mission` (stav a výsledek), výška a poloha modelu.
`mission.target_result` obsahuje WGS84 souřadnice, lokální souřadnice, identitu,
čas měření a odhad chyby. Je to poslední historické pozorování: po ztrátě cíle
zůstane uložené, ale jeho čas se neobnovuje a nepoužívá se pro další navádění.
Záměry rychlosti jsou v metrech za sekundu, sever/východ; `up_m_s` je kladné
nahoru a není přímo MAVLink NED vz. `flight_ready=false` znamená, že není
potvrzena způsobilost k reálnému letu, nikoliv výsledek simulace.

## Výpadky a převzetí

Ztracený cíl vyvolá brzdění a po 2 s návrat na nedokončený bod hledání.
Projetá trasa bez cíle skončí `SEARCH_COMPLETE` s nulovou vodorovnou rychlostí.
Omezení brzdné dráhy může dočasně vyvolat `ZONE_HOLD`.

Neplatný čas, stará telemetrie, ztráta mapy/kamery, režimu či armování,
odmítnutý povel, timeout nebo ztráta letové výšky vedou na trvalý `FAILSAFE`.
Ruční převzetí vede na trvalý `MANUAL` a záměr `RELEASE`; zastavení na `ABORTED`.
Tyto stavy se samy neobnoví. Nový běh vyžaduje novou instanci a opětovnou kontrolu
startu. `BRAKE` je požadavek backendu, nikoli důkaz zastavení; skutečné reakce
autopilota při výpadku spojení musejí být nakonfigurované na autopilotu.

## Rozhraní a kompatibilita

- `mission_controller.py`: nastavení, vstup, potvrzení, záměr a řadič bez I/O.
- `mission.py`: zachovaný import `Mission`; bez nastavení hlásí `DISABLED`.
- `mission_inputs.py`: převod `drone_data` do lokální startovní reference.
  Vyžaduje platnou živou detekci, odhad WGS84 a původní monotónní čas snímku.
  Starý JSON se nesmí omladit novým časem při čtení. Převod má limit 1 km.
- `mission_simulation.py`: backend s omezeným zrychlením a scénáři poruch.
- `tools/run_mission.py`: společný spouštěč simulace a zápis JSONL.

Starší `simulate_approach.py` a `run_sitl_approach.py` zůstávají samostatné testy
jednotlivých fází. Nejsou backendy nového řadiče. Produkční backend musí číst
potvrzení ze správného autopilota, převést jednorázové záměry na odpovídající
povely, zpracovat odmítnutí, ověřit režim a stav, pravidelně obnovovat rychlostní
povely a vždy respektovat `RELEASE`. Tento backend je v `mavlink_backend.py`.

## Skutečný let a páska

MAVLink backend (`mavlink_backend.py`), hlavní smyčka (`flight_runner.py`,
`tools/fly_mission.py`) a mapování pásky (`tape_mapper.py`) jsou hotové, viz
[FLIGHT.md](FLIGHT.md). Páska se promítá na zem a po dvou snímcích vznikne
zakázaná čára (`FieldMap.forbidden_lines`), kterou hlídá `ZoneGuard` i ořez
trasy. Simulace: `python3 tools/run_mission.py --tape N1 E1 N2 E2`.
Rychlost mise je omezena dohledem kamery na pásku (`tape_safe_speed`).

Stále chybí: ověření v ArduPilot SITL na vašem stroji, skutečné zapojení
a ladění prahů detekce na reálném poli.
