# Hackathon

Detektor terče „kolečko uvnitř obdélníku“ pro kameru dronu. Detekuje se jen
tato dvojice: samotné kolečko ani samotný obdélník nový cíl nezaloží.

```bash
/usr/bin/python3 main.py --picamera 0 --ev -2
```

Bez kamery: `python main.py --demo`, `python main.py --image fotka.jpg` nebo
`python main.py --video let.mp4` (záznam z reálného letu, na Pi např.
`rpicam-vid -t 30000 --codec libav -o let.mp4`). Q/Escape ukončí program.
Klávesy **1**, **2**, **3** přepínají citlivost hran, **B** vrací výchozí 1.5,
**E** přepíná obraz a hrany používané detektorem. Vyšší citlivost může přidat
falešné detekce. Náhled ukazuje čas zpracování v ms (bez času získání snímku
a přenosu přes VNC). Růžový nádech kamery se softwarově neodstraňuje.

## Detekce v jednom snímku

Čtyřúhelník: konvexní, uzavřený kontrastní obrys, plocha alespoň 500 px²,
strany alespoň 12 px, pravé úhly se nevyžadují (šikmý pohled). Obrys musí
polygon těsně sledovat, jinak by se za čtyřúhelník vydávala skvrna trávy.
Obrysy se hledají v hranách i v hranách s uzavřenými jednopixelovými mezerami.
Pokud nic nenajdou, záložní průchod spojí dlouhé slabé úsečky přes mezery v rozích.

Kolečko se hledá jen uvnitř masky čtyřúhelníku, s rezervou 2 px od okraje.
Při šikmém pohledu je kolečko elipsa; přijímá se poměr os od 0.35 (~70° od kolmého
pohledu). Obrysy pocházejí z hran i z Otsuova prahování (uzavřené i při slabém
kontrastu). Tvar ověřuje odchylka od elipsy v pixelech, pokrytí obvodu a 4. harmonická
poloměru, která odmítne i malý rozmazaný čtverec. Hough je záloha pro přerušené hrany
při téměř kolmém pohledu. Nerozlišuje význam objektu: kulatý odlesk v obdélníku
může být detekován stejně jako značka.

## Sledování mezi snímky (`src/tracker.py`)

Nový cíl vzniká jen z jediného kandidáta; po 4 navazujících detekcích je
**TERC POTVRZEN**. Při více kandidátech bez sledovaného cíle se nic nepotvrdí.
Potvrzený cíl drží i přes výpadky:

- až 10 snímků bez detekce: poloha se predikuje (oranžově, `measured = False`),
- až 15 snímků jen kolečko bez obdélníku na očekávaném místě (přeexponovaný okraj,
  terč u kraje obrazu); pak je obdélník opět nutný.

Jde o navazování blízkých poloh a velikostí, nikoli záruku identity objektu.
Na statickém obrázku/demu se opakuje tentýž snímek, stabilitu ověřte na videu
nebo živé kameře. Poloha je vyhlazená alfa-beta filtrem. `Observation.offset` dává odchylku cíle
od středu obrazu v rozsahu -1..1 (kladně vpravo a dolů) pro navedení nad střed.

## GPS poloha kolečka (`src/geolocation.py`)

Z pixelu kolečka, úhlů serv a polohy dronu spočítá paprsek a jeho průsečík se zemí:

```python
fix = locate((u, v), DronePose(lat, lon, height, roll, pitch, yaw),
             gimbal_from_servos(x_prikaz, y_prikaz, X_SERVO, Y_SERVO),
             CameraModel.load('camera_calibration.json'), frame_size=(640, 480))
fix.lat, fix.lon, fix.distance, fix.error  # GPS, vodorovná vzdálenost a chyba (1 sigma) v m
```

- Terén je rovný ve výšce místa startu. `height` je ArduPilot `relative_alt` (výška nad
  místem startu z EKF, výchozí zdroj je barometr), přesnější než samotná výška z GPS.
- Z ArduPilotu (MAVLink): `GLOBAL_POSITION_INT` (lat, lon, relative_alt) a `ATTITUDE`
  (roll, pitch, yaw). Serva nejsou stabilizovaná, náklon dronu se proto započítává.
- Závěs: vnější servo osa x (kladně = kamera kouká doprava) nese vnitřní servo osa y
  (kladně = dopředu), 0/0 = kolmo dolů. Při 0/0 míří horní okraj obrazu k přídi
  (jinak `image_top='right' | 'backward' | 'left'`).
- `centering_angles` vrátí úhly serv, které přesunou kolečko do středu obrazu (zámek).
- `combine` spojí opakovaná měření váženě podle chyby: měření nad kolečkem převáží šikmá.

Očekávaná chyba s výchozími nejistotami (serva ±2°, kompas ±3°, náklon ±1°, výška ±1 m,
GPS dronu ±1.5 m), v závorce vodorovná vzdálenost kolečka:

| výška | kolmo | 30° | 45° | 60° |
|---|---|---|---|---|
| 10 m | 1.6 m | 1.8 m (6 m) | 2.1 m (10 m) | 3.0 m (17 m) |
| 20 m | 1.9 m | 2.2 m (12 m) | 2.8 m (20 m) | 4.5 m (35 m) |
| 30 m | 2.2 m | 2.7 m (17 m) | 3.6 m (30 m) | 6.2 m (52 m) |

Šikmo tvoří největší část chyby serva, kolmo GPS dronu. Proto hrubý odhad ze středu
prostředí a zpřesnění přeletem nad kolečko.

Kalibrace před letem:

1. **Kamera:** vytiskněte šachovnici 10 x 7 polí (9 x 6 vnitřních rohů), změřte stranu pole
   a spusťte `python calibrate_camera.py --picamera 0 --square 25`. Vznikne
   `camera_calibration.json` (RMS by mělo být pod 0.5 px). Bez kalibrace se použije jmenovité
   zorné pole 53.5° x 41.4° (OV5647 + 3.6 mm) bez korekce zkreslení.
2. **Serva MG996R:** skutečný úhel neodpovídá příkazu přesně. Pro každé servo nastavte
   příkazy -60, -30, 0, 30, 60°, změřte skutečný sklon kamery (sklonoměr v telefonu na
   kameře) a proložte přímku: `ServoAxis(sign, scale, offset)`. Znaménko: kladný úhel x
   musí naklonit kameru doprava, kladný úhel y dopředu.
3. **Natočení obrazu:** při 0/0 dejte před příď předmět; musí být u horního okraje obrazu.
4. **Ustálení:** servo se otáčí asi 0.2 s na 60° a úhel nehlásí. Pro měření polohy
   berte snímky až po ustálení (alespoň 0.3 s po posledním pohybu) a v klidném visu.

## Testy

`python -m unittest discover -s tests -v`. `tests/synthetic_scene.py` vykresluje
terč na trávě perspektivní kamerou (náklon, rotace, výška, šum, rozmazání,
expozice, JPEG) i s návnadami: samotné kolečko, prázdný list, čtverec v listu,
samotný prstenec. Syntetická scéna neověřuje reálnou kameru: laďte na záznamech
z letu přes `--video`.
