# Páska a hlídání zóny

Náhled CSI kamery s kandidáty červenobílé pásky:

```bash
python3 main.py --picamera 0 --detect-boundary --status
```

Pouze data, případně uložit snímky pro doladění na skutečné pásce:

```bash
python3 main.py --picamera 0 --detect-boundary --headless --drone-data
python3 main.py --picamera 0 --detect-boundary --status --diagnostics logs/tape
```

Žluté úsečky v náhledu jsou kandidáti. Detektor hledá červenou a bílou barvu
s nejméně čtyřmi střídajícími se běhy podél přímky. Samotná tečka nebo jednobarevná
čára nestačí. Zpracování je omezeno na delší stranu 480 px a 64 kontrolovaných
úseček, výstup obsahuje nejvýše 16 kandidátů. Prahy jsou počáteční odhad pro
barevný obraz; růžový IR obraz, přesvětlení, malá vzdálenost mezi pruhy v pixelech,
ohnutá páska nebo zakrytí mohou rozpoznání znemožnit. Podobný vzor může způsobit
falešný nález. Ověřeno na syntetických obrazech, nikoliv na záznamu vašeho pole.

JSON `boundary` obsahuje `segments_px`, `state`, stáří a dobu zpracování.
`TAPE_CANDIDATES` znamená nalezený vzor, `NO_TAPE_OBSERVED` znamená, že se nenašel,
nikoliv že je prostor volný. `STALE` označuje starý snímek. Souřadnice úseček
jsou v původním obrazu. Tato detekce nenahrazuje detekci cílové tečky.

## Co skutečně hlídá zónu

`ZoneGuard` přijímá explicitně ověřený konvexní obvod v metrech, čas mapy,
polohu a rychlost vozidla a požadovanou rychlost. Vyžaduje známý čerstvý obvod
a telemetrii. Nepřijímá automaticky polygon získaný spojením konců viditelné pásky.
Kontroluje vzdálenost k hranám s rezervou 0.5 m, reakční dobou 0.3 s a brzdnou
dráhou podle maximální aktuální/požadované rychlosti a zrychlení 1 m/s².
Tyto hodnoty jsou parametry modelu, ne změřené schopnosti skutečného dronu.

Ve scénáři `--search` je guard zapojen před použitím vodorovného povelu;
při zamítnutí požaduje zabrzdění a resetuje regulátor. Mapa zde pochází ze
syntetického pole. Log obsahuje `zone_state`, `zone_source` a `boundary_clearance_m`.
Neznámá/stará mapa, neplatné údaje, nekonvexní polygon, poloha mimo obvod nebo
nedostatek místa k brzdění nepovolí pokračování. Nulový povel neznamená okamžité
zastavení ani návrat dovnitř, pokud už vozidlo hranici překročilo.

## Zbývající propojení pro skutečné pole

Živý výstup zůstává `ZONE_UNKNOWN`, `movement_allowed=false` a `flight_ready=false`.
Viditelná páska neurčuje uzavřený obvod ani jeho vnitřní stranu. Je nutné doplnit
časově sladěné promítnutí úseků na zem podle polohy dronu a kamery, slučování
pozorování do mapy, ověření uzavření a vnitřku podle startovní reference a ověřit
nejistotu mapy. Se servy bez zpětné vazby a nesynchronizovanou telemetrií nelze
současnou pixelovou detekci vydávat za spolehlivou letovou geofence.
`main.py` letové povely neposílá; automatické zmapování neznámého pole zatím hotové není.
