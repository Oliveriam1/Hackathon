# Struktura projektu po fázi 1

Spouštění a parametry zůstávají stejné. `main.py` načte konfiguraci a zavolá
`app.run`. Import modulů neotevírá kameru, GPIO ani MAVLink.

## Zapojené komponenty

| Modul | Odpovědnost |
| --- | --- |
| `config.py` | Typovaná konfigurace, CLI validace, dosavadní konstanty serv |
| `app.py` | Otevírání/zavírání prostředků, smyčka aplikace a obsluha chyb |
| `sources.py` | Výběr zdroje, demo, načtení obrázku a uložení fotografie |
| `camera.py`, `pi_camera.py` | Čtení snímků z konkrétního hardwaru nebo videa |
| `pipeline.py` | Jeden nový snímek → pozorování a datový záznam |
| `vision.py`, detektory, `tracker.py` | Detekce a sledování obrazu |
| `target_lock.py` | Platnost a centrování vizuálního měření |
| `mission.py` | Samostatný stav autonomie, zatím `NOT_IMPLEMENTED` |
| `preview.py` | Náhled, klávesy a ladicí posuvníky |
| `publisher.py`, `diagnostics.py` | JSON, čitelný stav a diagnostické snímky |
| `telemetry.py` | Čtení MAVLink telemetrie |
| `gimbal.py` | Zápis na serva; při zapnutí stále jen nájezd do 0/0 |

Datová cesta:

```text
zdroj → snímek + čas příjmu → pipeline → pozorování + JSON
                                        ↓              ↓
                                      náhled       výstup/diagnostika
```

`DetectionPipeline` neotevírá zařízení, nevytváří okna a netiskne data.
Závislosti na telemetrii a závěsu dostává při vytvoření. `ExitStack` v aplikaci
zavře otevřené prostředky i při chybě, Ctrl+C nebo konci záznamu.
Statický obrázek se nezpracovává opakovaně jako nová měření, dokud uživatel
nezmění citlivost. JSON nadále odděluje měření, predikci a stav vizuálního zámku.

## Připravené hranice dalších fází

Následující `Protocol` třídy jsou pouze smlouvy mezi komponentami,
nikoliv funkční implementace. Aplikace je zatím nevytváří ani nevolá.

| Modul | Budoucí vstup → výstup |
| --- | --- |
| `gimbal_controller.py` | Pozorování, úhly a čas → požadované úhly nebo žádná korekce |
| `boundary_detector.py` | Snímek a čas → úseky pásky v pixelech |
| `field.py` | Datový model ověřeného obvodu v metrech sever/východ od startu |
| `search_planner.py` | Mapa a poloha → další bod nebo žádný dostupný bod |
| `flight_controller.py` | Rozhraní pro vzlet, přesun a držení polohy |

Prázdná mapa znamená neznámý prostor. `StartReference` a `MissionConfig`
v konfiguraci připravují typy pro start a výšku vzletu, ale zatím nemají CLI
parametry ani řídicí účinek. Dosavadní `--altitude` je odhad pro velikost
tečky, nikoliv požadavek na vzlet. Matematika `geolocation.py` zůstává
nezapojená do hlavního zpracování.

Fáze 1 nemění detekční pravidla, řízení barev ani letové schopnosti.
`LOCKED` stále není potvrzení přeletu ani fyzického zaměření kamery.

## Ověření

```bash
python -m unittest discover -s tests -q
python main.py --demo --headless --status
python main.py --demo --detector geometry --headless
```

Testy používají syntetické obrazy a náhrady hardwaru. Pokrývají i oddělení
JSON od stavových výpisů, statický náhled, konec zdroje a úklid po chybách.
Neověřují fyzickou kameru, serva ani let.

## Doplnění: regulátor závěsu

`gimbal_controller.py` již obsahuje konkrétní časový regulátor. Pipeline ho
volá pouze při `--track-camera`; GPIO zápis zůstává v `gimbal.py`. Dry-run
nevytváří hardwarový závěs. Rozhraní plánovače a letového adaptéru zůstávají
neimplementovaná. Více viz DETECTION.md.
