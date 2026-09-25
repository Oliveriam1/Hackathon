# Přelet nad tečku – simulační fáze

Implementace je oddělená od `main.py`. Běžný program s kamerou nadále neřídí let
a `flight_ready` zůstává false. Simulace používá syntetický bod, ne živou detekci.

## Lokální simulace bez instalace autopilota

```bash
python3 tools/simulate_approach.py
python3 tools/simulate_approach.py --loss 3 7 --output trajectory.jsonl
python3 tools/simulate_approach.py --moving --seconds 40
```

Začíná již ve vzduchu, řeší vodorovný pohyb v rovině sever/východ. Cíl je
8 m na sever a 5 m na východ od začátku; pohyblivý scénář mění východní souřadnici.
Zóna ±20 m je syntetická obdélníková hranice, nikoliv rozpoznání pásky.

`approach.py` obsahuje regulátor: rychlost do 2 m/s, zrychlení 1 m/s², zpomalování
před cílem, potvrzení příletu při vzdálenosti do 0.4 m a rychlosti pod 0.15 m/s
po dobu 1 s. Při pohybu cíle obnoví přibližování. ARRIVED je tolerance v simulaci,
nikoliv důkaz přesného letu nad fyzickou tečkou.

Chybějící, neplatný, starý či příliš nejistý cíl vede na nulovou požadovanou
rychlost. Stejně tak stará telemetrie, výpadek řídicí smyčky a překročení modelu
zóny. Přílet se neurčuje podle středu kamery. Při nové identitě cíle se ruší
ustálení; predikce není náhrada čerstvé polohy cíle.

Kinematický model má konečné brzdění: nulový povel neznamená okamžité zastavení.
Kontrola zóny zahrnuje jednoduchou brzdnou dráhu. Nejde o ověřenou letovou geofence
ani o model větru, setrvačnosti a regulátorů skutečného dronu.

## Integrační test s ArduPilot SITL

Samostatný `tools/run_sitl_approach.py` je připraven pro lokální SITL. Potřebuje
nainstalované `pymavlink` a běžící simulovaný Copter na TCP 127.0.0.1:5760.
SITL musí být již ve vzduchu, armed a v GUIDED; skript sám nearmuje, nevzlétá
a nemění letový režim. Nepoužívejte přesměrování na skutečný autopilot.

```bash
python3 tools/run_sitl_approach.py --port 5760 --target 8 5 --seconds 30
```

Před jakýmkoliv pohybovým povelem ověří zprávy SIMSTATE a LOCAL_POSITION_NED
od stejného zdroje jako heartbeat ArduPilotu. Bez těchto dat končí. Sleduje
stáří polohy a heartbeat a ukončí test při opuštění armed GUIDED. Syntetický cíl
je relativní k lokální poloze při zahájení testu. Sériové porty ani vzdálené IP
tento spouštěč nepřijímá. Ověření SIMSTATE není obecný bezpečnostní mechanismus.

`sitl_velocity.py` překládá povel do SET_POSITION_TARGET_LOCAL_NED, rychlostního
režimu s nulovou svislou rychlostí. Formát vychází z
[oficiální dokumentace ArduPilot](https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html).
Povely je nutné obnovovat; spouštěč běží přibližně 20 Hz a při ukončení posílá
nulovou rychlost, pokud zůstal ve sledovaném GUIDED režimu.

## Co bylo ověřeno

- Lokální statický scénář dojel do ARRIVED, přibližně 0.20 m od tečky.
- Výpadek cíle v čase 3–7 s vyvolal brzdění, zastavení a následný návrat k cíli.
- Pohyblivý scénář skončil po 40 s přibližně 0.22 m od tečky.
- Testy kontrolují rychlost, zónu, brzdění, stáří, nejistotu a kódování MAVLink.

ArduPilot SITL zde není nainstalovaný, WSL také ne a lokální Python nemá pymavlink.
Skutečný běh integračního testu SITL proto **nebyl proveden**. Prošel kinematický
model a testy s náhradou MAVLink spojení; nelze je vydávat za letový test.

Další krok po SITL: propojit časově sladěná měření lokalizace s polohou vozidla
ve stejné souřadné soustavě. Následně ověřit hranice, reakce na výpadky a chování
celé sestavy. Živý přenos z kamery do letového řízení zatím není zapojený.
