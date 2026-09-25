# Hackathon

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

Pro malé terče se analyzuje dvakrát zvětšený výřez (kratší strana pod 80 px),
nikoli celý zvětšený obraz. Souřadnice se převádějí zpět do původního rozlišení.
Minimální plocha čtyřúhelníku je 200 px² a strana 8 px. Vyšší rozlišení může
zlepšit dosah za cenu času zpracování; zvětšení samo neobnoví chybějící detaily.
Ověřeno synteticky na terčích šířky 24, 32 a 48 px, nikoli jako garantovaný dosah v metrech.

Tok programu: `parse_args` → kamera/obraz → `Vision.observe` → `detection_record`
→ `JSONPublisher` → volitelná vizualizace. Kamera a soubor se zavírají přes ExitStack.
Režim `--snapshot` zůstává samostatným uložením surové fotografie bez detekčních dat.
Testy této větve: `python -m unittest discover -s tests -v`.

## Červené tečky podle red_tracker.py

Spuštění na CSI OV5647 NoIR se stejným rozlišením a profilem jako v dodaném skriptu:

```bash
python main.py --picamera 0 --detector red --width 1296 --height 972 --headless --status
```

Pro JSON vynechte `--status`. Pro náhled vynechte `--headless`; E zobrazí červenou masku.
Režim red automaticky načítá `ov5647_noir.json`. Tento profil upravuje zpracování barev,
nevypíná IR LED. `--tuning-file none` použije systémový profil, jinak lze zadat vlastní soubor.
Chyba načtení profilu se hlásí, nepřepíná se potichu na jinou kameru.

Detektor přebírá rozdíl R-max(G,B)>25, R>50, relativní převahu červené >0.35,
uzavření masky 3x3 a vážený střed komponenty. Nevyžaduje okolní čtyřúhelník.
Jde o detekci červených oblastí, nikoliv záruku kruhového tvaru: červený čtverec může také projít.
Vrací všechny kandidáty; více kandidátů brání počátečnímu potvrzení cíle.
Volitelně `--red-diameter-px 10` filtruje velikost na 0.4 až 2.5 násobek zadaného průměru.
Bez tohoto parametru je filtr velikosti vypnutý, protože nemáme ověřenou výšku a geometrii.
Průměr je v pixelech aktuálního rozlišení. `--sensitivity` a `--hough` patří geometrické detekci.

Původní režim zůstává dostupný jako `--detector geometry` (výchozí).
Houghova záloha je nově volitelná přes `--hough`; výchozí geometrická detekce používá obrysy a Otsu.
Status obsahuje časy fází v ms (předzpracování/čtyřúhelníky/kolečka a tracking),
v červeném režimu jeden čas `red`. JSON obsahuje `detector_mode` a `stage_ms`.

Z dodaného skriptu se nepřebírá pevná výška 20 m, poloha dronu, rozměry pole ani GPIO serv.
Stávající výstupy měření, predikce a visual_lock zůstávají; geolokace není dostupná
bez skutečných vstupů. Tato integrace neposílá letové ani servo povely.

## Struktura po fázi 1

Rozdělení komponent a rozhraní dalších fází popisuje [ARCHITECTURE.md](ARCHITECTURE.md).
`main.py` nyní pouze načte konfiguraci a spustí aplikaci; stejné příkazy fungují dál.
Detekce a nastavení kamery zůstávají beze změny. Letová mise a automatické
sledování servy zatím implementované nejsou.
