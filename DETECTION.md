# Detekce červené tečky, tracking a výstup pro řízení

## Pravidla a omezení

- Červená a růžová maska mají společné ověření komponent. Růžová neobchází filtr velikosti.
- Kontroluje se izolace a barevný kontrast vůči lokálnímu okolí. Větší komponenty musí
  odpovídat elipse; malé komponenty pod 10 px mají tvar označený jako nerozlišitelný.
  Neexistuje záruka, že několikapixelový červený objekt je kruh.
- Žádné plošné uzavření 3×3: sousední malé cíle se zbytečně neslepují.
- `--red-diameter-px` je explicitní tvrdý filtr 0.4–2.5 násobku průměru.
  Ruční výška, relativní výška autopilota a nominální zorný úhel jsou pouze odhad
  pro skóre. Chybějící výška zůstává neznámá. Relativní výška není měřená vzdálenost k terči.
- `--camera-calibration camera.json` využívá ohniskovou vzdálenost z `CameraModel`.
  Rozlišení musí odpovídat. Měřítko je stále přibližné (rovina terče, náklon,
  poloha v obrazu), nikoliv výpočet světových souřadnic.

Detektor vrací všechny kandidáty se skóre. `TimedTargetTracker` vlastní identitu,
rychlost v px/s a oblast hledání. Geometrický režim používá původní tracker.
Podobné skóre při pořízení nebo asociaci vede na `AMBIGUOUS`, nikoliv nové měření.
Potvrzení vyžaduje nejméně čtyři měření a 0.25 s. Delší mezery potvrzování přeruší.
Krátký výpadek může zachovat identitu, ale výstup je predikce. Po 0.7 s bez měření
potvrzený cíl zaniká; další cíl musí být znovu potvrzen a dostane nové ID.
ID je lokální pro běh trackeru, nikoliv globální identifikátor skutečného objektu.
Při překrytí vizuálně stejných objektů není zachování identity zaručeno.

Oblast hledání sleduje predikci a roste s pohybem a nejistotou. Pravidelně se prohledá
celý snímek; při prázdném nebo oříznutém výřezu se hledání rozšíří ihned.
Nečekaný velký přesun kamery může způsobit ztrátu a nové potvrzení, nikoliv slepý
přenos identity na vzdálenou skvrnu. `--geometry-debug` je volitelná analýza okolí
kandidáta a nerozhoduje o platnosti cíle. Náhled již znovu nespouští segmentaci.

## Diagnostika

`--status` ukazuje stav trackingu a počty zamítnutí. Fáze v ms jsou:
**maska / komponenty / ověření / geometrie / tracking**.
Plný JSON přidává:

- `detection_diagnostics.scale`: očekávaný průměr, původ výšky, tvrdý/měkký filtr;
- `detection_diagnostics.roi_px`: aktuální výřez, `null` = celý snímek;
- `detection_diagnostics.candidates`: přijetí, důvod, oblast, střed, kontrast a skóre;
- `tracking`: stav, ID, rychlost a stáří posledního měření;
- `stage_ms`: časy jednotlivých částí a celku.

Seznam diagnostických komponent je omezen na 100 záznamů. Příznak `reports_truncated`
označuje zkrácení; samotné hledání a počet kandidátů se neomezují. Jedno- a dvoupixelové
komponenty se počítají souhrnně jako `TOO_SMALL`. Skóre není kalibrovaná pravděpodobnost.

## Kontrakt pro budoucí letový regulátor

```bash
python3 main.py --picamera 0 --width 1296 --height 972 --drone-data
```

Každý řádek je JSON `message_type: vision_target`. Důležitá pole:

| Pole | Význam |
| --- | --- |
| `measurement_valid` | Potvrzené měření s platným vizuálním zámkem a kontrolou stáří |
| `measurement_age_ms` | Čas od načtení snímku do vyhodnocení, nikoliv stáří expozice |
| `state` | SEARCHING, LOCKED, IMAGE_CENTERED, AMBIGUOUS, LOST nebo STALE_OR_REPEATED |
| `target_id` | Identita trackeru, může existovat i při dočasné ztrátě |
| `target_px` | Aktuální změřený střed; `null` při neplatném měření |
| `camera_error_normalized` | Odchylka od středu / polovina rozměru obrazu; +x doprava, +y dolů |
| `image_centered` | Ustálené vycentrování obrazu; neznamená, že dron je nad cílem |
| `input_source` | camera, video nebo static; přehrávání není živý letový vstup |
| `world_position` | Zatím `null` |
| `flight_ready`, `flight_command` | Zatím `false`, `null` |

Tento formát se vypisuje do stdout nebo do `--output`. Není automaticky odesílán
autopilotovi. Souřadnice obrazu nelze bez orientace kamery použít jako směr letu.

## Záznam a opakovatelné vyhodnocení

```bash
python3 main.py --picamera 0 --width 1296 --height 972 --headless --status --record-dir recordings --frames 150
```

Vznikne nová složka `recordings/sequence-...` s originálními PNG a `manifest.jsonl`.
Zápis snímků může snížit FPS; časování skutečného běhu měřte také bez záznamu.
Nahrajte samostatné sekvence: terč v různých vzdálenostech, prázdné pole, pásku,
stín, více teček, pohyb kamery a zakrytí cíle.

V každém řádku manifestu doplňte `targets`. Například:

```json
{"image":"000000.png","time_s":0.0,"targets":[{"id":"terc","x":320,"y":240,"radius":6}]}
```

Prázdný seznam `[]` znamená snímek bez cíle. `null` znamená neanotováno a hodnoticí
nástroj ho odmítne. Anotace musí označit všechny skutečné cíle, ne jen detekované.
Při spojení sekvencí lze přidat pole `sequence`; při změně se tracker resetuje.

```bash
python3 tools/evaluate_detection.py recordings/sequence-.../manifest.jsonl --output report.json
```

Report obsahuje precision, recall, falešně potvrzené snímky, změny identity,
chybu středu, medián a p95 zpracování. Nejde o čas čtení kamery, GUI nebo zápisu.
Porovnávejte stejnou sadu při stejném rozlišení a na stejném zařízení. Přísnější
filtry přijmeme jen s kontrolou, že nezmizely malé nebo šikmo viděné cíle.

## Dosavadní ověření

Regresní testy zahrnují čtverec, pruh, velkou růžovou oblast, malé a šikmé cíle,
šum/stín, nejednoznačnost, různé FPS, výpadek, nové ID, oříznutý výřez,
stará/opakovaná měření, zdroj měřítka, záznam a vyhodnocení.

Lokální orientační porovnání na syntetickém snímku 1296×972 s šedými rámečky
a jednou červenou tečkou (22 měření po zahřátí): plné hledání původního detektoru
medián 25.12 ms, nového 18.23 ms; p95 27.82 vs 19.18 ms. Původní algoritmus
byl načten z tehdejšího HEAD. Výsledek není měřením Raspberry Pi ani přesnosti v poli.
