# Hackathon

## Skutečný let (nové)

`tools/fly_mission.py` spojuje kameru, pásku jako zakázanou zónu, řadič mise a
ArduPilot. Zapojení, parametry autopilota a postup testování: [FLIGHT.md](FLIGHT.md).

```bash
python3 tools/fly_mission.py --connect /dev/serial0 --baud 921600 --real-flight --field-bounds -10 10 -10 10
python3 tools/run_mission.py --tape 5 -12 5 12   # simulace s páskou
```

## Ověření bez zasekaného živého okna

```bash
/usr/bin/python3 main.py --picamera 0 --headless --status --diagnostics kontrola
```

Stav v terminálu 1x/s: FPS celého cyklu, čas detekce, počet čtyřúhelníků/kandidátů,
aktuální měření nebo predikce a stav locku. Nejde o naměřené FPS samotného senzoru.
`--output detections.jsonl` současně uloží strojová data. `--sensitivity 1` nastaví
přísnější prahy i bez grafického okna; výchozí je 1.5.

Ve složce kontrola/run-* vzniknou každé dvě sekundy raw PNG, marked PNG a JSON
stejného snímku, nejvýše 30 sad. Ukládání může krátce zpomalit cyklus; pro čisté
měření výkonu spusťte bez --diagnostics. Každý běh používá vlastní podadresář.

Test: 20 s scéna bez terče (nesmí hlásit lock), pak známý terč, pohyb a zakrytí.
Zelený overlay označuje potvrzení detektorem, nikoli důkaz správné klasifikace.
Při zakrytí smí tracker krátce ukazovat PREDIKCE, ale vizuální lock musí být LOST
a nesmí poskytovat platné měření pro řízení. Je-li detekce nad 250 ms,
lock hlásí STALE_OR_REPEATED. Pro rozbor falešného nálezu použijte odpovídající
raw/marked snímky. Syntetické testy nezaručují správnost na skutečné scéně.

## Vizuální lock

Každý JSON nyní obsahuje `visual_lock`: stav SEARCHING / LOCKED / IMAGE_CENTERED /
LOST / STALE_OR_REPEATED, normalizovanou chybu obrazu a stáří měření.
X je kladně doprava, Y dolů. Jde o souřadnice **obrazu**, ne povely pro osy dronu.
Lock používá pouze potvrzené aktuální měření, nikdy trackerovou predikci.
Vzorek starší než 250 ms od přijetí nebo opakovaný čas se odmítá.
IMAGE_CENTERED vyžaduje alespoň 0,5 s navazujících měření kolem středu s hysterezí
3 % / 5 % poloviny rozměru obrazu. Ztráta nebo dlouhá mezera tento čas vynuluje.

Kamera je na pohyblivých servech: IMAGE_CENTERED tedy znamená pouze to, že kamera
míří na terč. Není to potvrzení polohy dronu nad ním. Změna úhlů serv mění
převod obrazu do souřadnic těla dronu, proto zatím `flight_command=null`.
Fyzické natáčení čeká na známé zapojení serv (GPIO/PCA9685, piny/kanály), kalibraci
jejich rozsahů a orientaci obrazu. Řízení letu navíc potřebuje připojení autopilota
a omezení povolené oblasti. Pole lock není watchdog: příjemce musí sám odmítat
staré zprávy, pokud kamera nebo celý program přestane vysílat.

## Živá telemetrie ArduPilotu

Volitelně přidejte `--mavlink PORT --baud RYCHLOST`. Například **jen pokud
váš kontrolér skutečně používá tento port**:

```bash
/usr/bin/python3 main.py --picamera 0 --headless --mavlink /dev/ttyACM0 --baud 115200
```

`pymavlink` musí být dostupný ve stejném Python prostředí. `--target-system 1`
omezí spojení na dané MAVLink system ID. Po spojení si adaptér vyžádá
GLOBAL_POSITION_INT (10 Hz), ATTITUDE (30 Hz), GPS_RAW_INT (2 Hz).
Průběžně vysílá heartbeat palubního počítače. Požadované frekvence nejsou
zaručené; chybějící/zastaralé zprávy jsou viditelné v JSON.
Postup odpovídá [rozhraní ArduPilot MAVLink](https://en.ardupilot.org/dev/docs/mavlink-commands.html).

JSON `telemetry` obsahuje data, stáří jednotlivých zpráv a stav fresh/missing/stale/error.
`relative_alt_m` je výška vůči home, nikoliv měřená vzdálenost od země.
Časy jsou časy přijetí a nejsou synchronizované s expozicí kamery.
Aktuálně se nepřepíná letový režim, nearmuje a nevysílají letové povely.
`flight_ready=false` a `autonomy.enabled=false` výslovně znamenají, že připojení
telemetrie samo o sobě nezprovozňuje autonomní let.

Pro autonomní misi zbývá: ověřený letový adaptér, určení povolené oblasti,
kalibrace a časované úhly serv, synchronizace expozice/telemetrie, vyhledávací
trajektorie a reakce na ztrátu cíle/spojení. Přítomnost GPS fixu není náhradou
za EKF/pre-arm kontroly. Program zatím pásku jako hranici nepoznává.

## Data z detekce (větev main)

Bez okna, výsledky jako JSON Lines do terminálu (Ctrl+C ukončí sběr):

```bash
/usr/bin/python3 main.py --picamera 0 --headless --width 1280 --height 720
```

Ukládání do souboru (záznamy se připojují, soubor se nemaže):

```bash
/usr/bin/python3 main.py --picamera 0 --headless --width 1280 --height 720 --output detections.jsonl
```

Vynecháním `--headless` zapnete diagnostický náhled, data se zapisují i s náhledem.
`--frames 100` omezí počet zpracovaných snímků. `--ev -2` je volitelné ztmavení CSI kamery.
`--demo --headless` nebo `--image fotka.jpg --headless` vrátí jeden záznam:
opakování stejné fotografie nesmí vytvořit falešné potvrzení přes několik snímků.

Každý záznam má číslo snímku, čas přijetí, rozlišení, počet kandidátů,
stav potvrzení, `measurement_px` (aktuální nezhlazené měření) a
`tracked_position_px` (vyhlazenou polohu či predikci). `position_kind` tyto
stavy rozlišuje. Pro další výpočet používejte `measurement_px` pouze pokud
`valid_pixel_position=true`. Při výpadku jsou měření i odchylka `null`, i když
tracker ještě krátce drží potvrzený odhad. Poloměry jsou poloosy elipsy v pixelech.

`received_at_unix_s` je čas na počítači po načtení snímku, nikoli hardwarový čas
expozice; u videa nejde o původní čas natáčení. Číslování začíná při spuštění od 1.
`geolocation=null`: main zatím nemá připojenou telemetrii ani kalibraci pro výpočet
zeměpisných souřadnic. JSON není automaticky posílán na notebook přes síť.

V režimu `geometry` se pro malé terče analyzuje dvakrát zvětšený výřez (kratší strana pod 80 px),
nikoli celý zvětšený obraz. Souřadnice se převádějí zpět do původního rozlišení.
Minimální plocha čtyřúhelníku je 200 px² a strana 8 px. Vyšší rozlišení může
zlepšit dosah za cenu času zpracování; zvětšení samo neobnoví chybějící detaily.
Ověřeno synteticky na terčích šířky 24, 32 a 48 px, nikoli jako garantovaný dosah v metrech.

Tok programu: `parse_args` → kamera/obraz → `Vision.observe` → `detection_record`
→ `JSONPublisher` → volitelná vizualizace. Kamera a soubor se zavírají přes ExitStack.
Režim `--snapshot` zůstává samostatným uložením surové fotografie bez detekčních dat.
Testy této větve: `python -m unittest discover -s tests -v`.

## Červené tečky a tracking

```bash
python3 main.py --picamera 0 --detector red --width 1296 --height 972 --status
```

Výchozí režim je `red`. Bez `--status` vypisuje plný JSON; `--headless` vypne okno.
Klávesa E přepíná masku. Profil `ov5647_noir.json` nevypíná IR LED.
Nastavení CSI kamery zůstává zachované. `--tuning-file none` použije systémový profil.

Červená i růžová procházejí společným ověřením velikosti, tvaru a místního kontrastu.
Větší tečky musí mít eliptický obrys; u několika pixelů nelze tvar spolehlivě určit.
Detektor vrací všechny kandidáty. Červený tracker používá čas, predikci a adaptivní
výřez; podobně pravděpodobné cíle hlásí jako `AMBIGUOUS` bez nového platného měření.
Geometrická detekce celého terče zůstává dostupná přes `--detector geometry`.

`--red-diameter-px 13` je tvrdý filtr velikosti pro obě barvy.
`--altitude`, případně čerstvá relativní výška z telemetrie, jen upravuje skóre.
Bez výšky je měřítko neznámé; není dosazena výška 20 m.
Konfigurovat lze také `--target-diameter-m`, `--hfov-deg` a `--camera-calibration`.
`--geometry-debug` zapne doplňkovou geometrii okolí kandidáta.

## Data pro řídicí část

Náhled kamery a současně pouze datový kontrakt pro budoucí řízení:

```bash
python3 main.py --picamera 0 --detector red --width 1296 --height 972 --drone-data
```

Přidáním `--headless` zůstanou jen data. `--drone-data` nekombinujte s `--status`.
`--output drone.jsonl` zapíše stejná data do souboru místo terminálu.
JSON obsahuje platnost/stáří měření, stav, identitu, pixelovou polohu a normalizovanou
odchylku v souřadnicích obrazu. Neplatná, stará či nejednoznačná měření mají cílovou
polohu `null`. Plný běžný JSON obsahuje tento kontrakt v položce `drone_data`.

`flight_ready=false`, `flight_command=null`, `world_position=null`: datový výpis
zatím neodesílá letové povely ani neposkytuje polohu cíle na zemi.
Podrobnosti a opakovatelné testování: [DETECTION.md](DETECTION.md).

## Struktura po fázi 1

Rozdělení komponent a rozhraní dalších fází popisuje [ARCHITECTURE.md](ARCHITECTURE.md).
`main.py` nyní pouze načte konfiguraci a spustí aplikaci; stejné příkazy fungují dál.
Následné změny detekce popisuje DETECTION.md; nastavení kamery zůstává zachované. Letová mise a automatické
sledování servy zatím implementované nejsou.

## Sledování tečky kamerou

Regulátor je implementovaný. Pro neověřené zapojení použijte výpočet bez pohybu:

```bash
python3 main.py --picamera 0 --width 1296 --height 972 --track-camera --gimbal-dry-run --status
```

Detaily zapojení, geometrických předpokladů a aktivace jsou v [DETECTION.md](DETECTION.md).

## Lokalizace tečky

Výpočet GPS a posunu cíle na zemi je nyní zapojen přes `--locate-target`.
Vyžaduje telemetrii, známou rovinu terče a úhly kamery; nejde o letový povel.
Konfiguraci a omezení odhadu popisuje [DETECTION.md](DETECTION.md#poloha-tečky-na-zemi).

## Simulace přeletu nad tečku

```bash
python3 tools/simulate_approach.py --loss 3 7
```

Samostatná simulace ověřuje přiblížení, zpomalení, zastavení při ztrátě cíle a návrat.
Není napojená na skutečný dron. SITL spouštěč, omezení a výsledky viz [SIMULATION.md](SIMULATION.md).

## Celá mise – simulační řadič

`python3 tools/run_mission.py --seconds 180` spustí společný řadič od kontroly
startu přes vzlet a hledání po držení cíle a výstup souřadnic. Podrobný kontrakt,
scénáře poruch a dosud chybějící části skutečného letu jsou v [MISSION.md](MISSION.md).
