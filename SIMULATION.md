# Přelet nad tečku – simulační fáze

Implementace je oddělená od `main.py`. Běžný program s kamerou nadále neřídí let
a `flight_ready` zůstává false. Simulace používá syntetický bod, ne živou detekci.

## Lokální simulace bez instalace autopilota

```bash
python3 tools/simulate_approach.py
python3 tools/simulate_approach.py --loss 3 7 --output trajectory.jsonl
python3 tools/simulate_approach.py --moving --seconds 40
python3 tools/simulate_approach.py --takeoff-height 5 --seconds 40 --output takeoff.jsonl
```

Bez `--takeoff-height` začíná již ve vzduchu a řeší vodorovný pohyb v rovině sever/východ. Cíl je
8 m na sever a 5 m na východ od začátku; pohyblivý scénář mění východní souřadnici.
Zóna ±20 m je syntetická obdélníková hranice, nikoliv rozpoznání pásky.

S `--takeoff-height 5` začíná na zemi: `TAKING_OFF` → `TAKEOFF_SETTLING` →
`AIRBORNE` → přelet nad bod. Výška musí být 1–20 m. Přelet se povolí po celé
sekundě ve výškové toleranci ±0.2 m a při svislé rychlosti pod 0.15 m/s.
Do té doby jsou vodorovné povely nulové. Stoupání v modelu je omezené na 1 m/s,
svislé zrychlení na 1 m/s². Výstup obsahuje výšku, svislou rychlost a
`takeoff_complete`. `takeoff.py` ověřuje čerstvost dat, souvislost měření
a časový limit 60 s; chyba zůstane aktivní do nového spuštění.

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

## Hledání po poli v lokálním modelu

```bash
python3 tools/simulate_approach.py --search --takeoff-height 5 --seconds 300 --output search.jsonl
python3 tools/simulate_approach.py --search --no-target --seconds 300
```

`RectangleSweep` vytváří střídavé řádky v explicitním obdélníku s rezervou
1 m od hran. Výchozí pole je sever/východ −10 až +10 m; lze změnit přes
`--field-bounds N_MIN N_MAX E_MIN E_MAX`. Start (0, 0) musí ležet uvnitř.
`--lane-spacing` nastavuje největší rozestup řádků (výchozí 4 m).
Prázdná či neplatná hranice není povolený prostor. Tento plánovač nepřijímá
obecný polygon a neumí získat hranice z červenobílé pásky.

`SearchMission` po vzletu postupuje po bodech trasy (`SEARCHING`). Poloha cíle
se jí zpřístupní až v ideálním kruhovém dosahu `--sensor-radius` (výchozí 3 m).
Tento model nenahrazuje kameru, výpočet zorného pole ani potvrzování detekcí.
Při čerstvém platném měření přejde na přelet a ustálení nad cílem. Při ztrátě
požaduje nulovou rychlost (`TARGET_LOST_HOLD`) a po 2 s obnoví původní bod trasy.
Po projetí všech bodů bez cíle zůstane stát (`SEARCH_COMPLETE`). Dokončení
trasy není zárukou detekce všeho: dosah senzoru, rozestup, okraje a pohyb cíle
mohou způsobit nepozorované oblasti. Výstup uvádí `target_visible`,
`search_waypoint` (index od nuly) a `search_waypoint_count`; skutečná poloha
syntetického cíle je v logu pouze pro vyhodnocení.

Hledání je zatím zapojené pouze do lokální kinematické simulace. SITL spouštěč
nadále testuje přelet k předem zadanému bodu. `main.py` ani živou kameru nový
plánovač neřídí.

V režimu hledání nyní každý povel kontroluje také `ZoneGuard` s rezervou a
brzdnou dráhou. Mapa je explicitně označená `synthetic_field`. Pixelové
rozpoznávání pásky v kamerové aplikaci a jeho omezení popisuje [BOUNDARY.md](BOUNDARY.md).

## Integrační test s ArduPilot SITL

Samostatný `tools/run_sitl_approach.py` je připraven pro lokální SITL. Potřebuje
nainstalované `pymavlink` a běžící simulovaný Copter na TCP 127.0.0.1:5760.
Bez `--takeoff-height` musí být SITL již ve vzduchu, armed a v GUIDED.
Nepoužívejte přesměrování na skutečný autopilot.

```bash
python3 tools/run_sitl_approach.py --port 5760 --target 8 5 --seconds 30
python3 tools/run_sitl_approach.py --takeoff-height 5 --target 8 5 --seconds 30
```

Volitelný automatický vzlet vyžaduje nearmovaný Copter na zemi u home.
`sitl_takeoff.py` nastaví GUIDED, běžně armuje (bez force) a pošle TAKEOFF.
Každý povel vyžaduje přijatý COMMAND_ACK a odpovídající stav telemetrie;
odmítnutí nebo timeout test ukončí. Samotné přijetí TAKEOFF nestačí:
přelet začne až po potvrzení ustálené výšky z GLOBAL_POSITION_INT.relative_alt
(nad home, nikoliv terénem). Omezení rychlosti vzletu v SITL určuje autopilot.
Při chybě skript neopakuje armování ani automaticky nelanduje/disarmuje;
stav simulátoru je nutné zkontrolovat v jeho konzoli. `--seconds` v SITL měří
až dobu přeletu po vzletu, v lokálním modelu celou simulaci.

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
