# UAV Vision

Python 3.11+; detekce zeleného referenčního bodu a červených objektů přes HSV.
Geolokace pouze počítá souřadnice. Projekt neposílá žádné letové příkazy.

## Spuštění

```sh
python main.py --dry-run
python main.py --debug
python -m unittest discover -v
```

Na macOS použijte Python z `.venv` a závislosti z `requirements.txt`
(`opencv-python` se importuje jako `cv2`). Na Raspberry Pi lze OpenCV instalovat
systémově jako `python3-opencv` a použít `requirements-pi.txt`; venv pak musí mít
přístup k systémovým balíčkům. Žádné nové závislosti nebyly přidány.

Volitelný lokální náhled přes OpenCV (ukončení `q`, Escape nebo zavřením okna):

```sh
python main.py --camera --camera-index 0
python main.py --camera --calibration camera-calibration.json
```

Bez kalibrace se zobrazí `CAMERA CALIBRATION REQUIRED`, bez telemetrie
`GEOLOCATION: NO TELEMETRY`. Detekce barev stále funguje. Neshoda rozlišení
s kalibrací zablokuje pouze geolokaci. `--dry-run` nikdy neotevírá kameru, ani
v kombinaci s `--camera`. Běžný import ani spuštění bez `--camera` hardware neotevírá.
Backend podporuje kamery dostupné přes OpenCV; samostatný backend Pi CSI/Picamera2
zatím není implementovaný. Náhled potřebuje grafické prostředí.

## Odpovědnosti modulů

- `detector.py`: zelená/červená detekce, obrazové souřadnice a odchylka zeleného bodu.
- `camera.py`: načtení kalibrace a explicitně otevřený kamerový backend.
- `telemetry.py`: protokol `TelemetryProvider` a výslovně zadaná statická testovací data.
- `geolocation.py`: paprsek kamery, průsečík se zemí a lokální WGS84 převod.
- `field.py`: volitelný perspektivní grid pro debug; není zdrojem GNSS souřadnic.
- `publisher.py`: zatím prázdný, nic se automaticky neodesílá.

Původní grid a jeho testy zůstávají zachované. Pro GNSS geolokaci nejsou potřeba
rohy soutěžního pole ani `FieldGrid`.

## Kalibrace kamery

Jediným místem konfigurace kamery pro CLI je JSON předaný přes `--calibration`.
Soubor si vytvořte z naměřené kalibrace. Následující šablona je **záměrně neplatná**:
všechny `null` je nutné nahradit skutečnými hodnotami. Aplikace nemá výchozí FOV
ani výchozí montáž kamery.

```json
{
  "intrinsics": {
    "width": null,
    "height": null,
    "fx": null,
    "fy": null,
    "cx": null,
    "cy": null,
    "distortion_coefficients": null
  },
  "mount": {
    "roll_deg": null,
    "pitch_deg": null,
    "yaw_deg": null
  }
}
```

`fx`, `fy`, `cx`, `cy` jsou v pixelech pro konkrétní rozlišení a orientaci snímku.
Při změně velikosti, výřezu, rotaci či zrcadlení musí odpovídat také kalibrace.
`distortion_coefficients` může zůstat `null`, pokud koeficienty nejsou dostupné;
pak se používá ideální pinhole model. Jinak jde o 4, 5, 8, 12 nebo 14 koeficientů
ve standardním pořadí OpenCV (nikoli fisheye model).

Pro známé FOV je dostupná aproximace
`CameraIntrinsics.from_fov(width, height, horizontal_fov_deg, vertical_fov_deg)`.
Obě FOV je třeba dodat; tato utilita nenahrazuje skutečnou kalibraci objektivu.

## Souřadné soustavy a montáž

- Kamera OpenCV: X doprava, Y dolů, Z opticky dopředu.
- Tělo UAV FRD: X dopředu, Y doprava, Z dolů.
- Svět NED: X sever, Y východ, Z dolů.

Používáme sloupcové vektory a aktivní pravotočivé rotace
`Rz(yaw) @ Ry(pitch) @ Rx(roll)`. Vnější API má úhly ve stupních.
UAV s nulovou orientací míří dopředu na sever. Kladný roll sklápí pravé křídlo,
kladný pitch zvedá příď, kladný yaw otáčí ze severu na východ.
Yaw musí být vůči **skutečnému severu**, ne nekorigovanému magnetickému severu.

Nulový `CameraMount(0, 0, 0)` znamená kameru mířící dopředu:
kamera Right → tělo Right, kamera Down → tělo Down, kamera Forward → tělo Forward.
Na tuto základnu se aplikuje mount rotace `Rz @ Ry @ Rx` v soustavě FRD.
Například **výslovně zvolený** `CameraMount(0, -90, 0)` znamená kameru dolů
s horní hranou obrazu směrem dopředu. Není to produkční default.
Při nulové orientaci UAV pak pixel vpravo znamená východ, pixel nahoře sever.

## Telemetrie a výpočet

`UAVTelemetry` vyžaduje WGS84 latitude/longitude, `altitude_agl_m` a
`Attitude(roll_deg, pitch_deg, yaw_deg)`. **AGL je výška nad místní rovinou země.**
GNSS výška nad elipsoidem, výška nad mořem ani výška vůči startu se automaticky
nesmí použít jako AGL. Je potřeba měření výšky nad terénem nebo správně odvozené AGL.

`TelemetryProvider.get_telemetry()` je rozhraní nezávislé na autopilotu.
Skutečný adaptér dosud není implementován; musí převést jednotky a soustavy,
odmítnout neplatná/zastaralá data a zajistit telemetrii v čase expozice snímku.
Budoucí adaptér lze předat do `run_camera(..., telemetry_provider=provider)`.

Pouze pro demonstraci lze přidat `--mock-telemetry`: použije statické testovací
hodnoty 50° N, 14° E, AGL 10 m, orientaci 0/0/0. Log i obraz výslovně označují
`STATIC TEST TELEMETRY`; výsledky nejsou skutečné souřadnice pozorovaných objektů.
Žádná kalibrace se tím nedoplňuje.

Pro každý červený objekt:

```python
geolocator = TargetGeolocator(intrinsics, camera_mount)
for detection in red_detections:
    location = geolocator.geolocate(detection.center, telemetry)
```

Postup: pixel → případné odstranění zkreslení → jednotkový paprsek přes K⁻¹
→ camera-to-body → body-to-NED → průsečík s `z = altitude_agl_m`
→ North/East offset → WGS84 latitude/longitude. Zelený bod slouží pouze
k výpočtu `error_x`, `error_y`, `centered`.

Paprsek nad horizontem nebo téměř rovnoběžný se zemí vyvolá `ValueError`.
V debug režimu se označí pouze tento objekt a zpracování dalších pokračuje.
WGS84 převod používá místní meridiánový a příčný poloměr křivosti, nikoli pevný
počet kilometrů na stupeň. Je určen pro metry až desítky metrů; tato implementace
odmítá offsety nad 1 km a nenulové offsety do 1° od pólů. Nejde o globální geodetický výpočet.

## Přesnost a reálné UAV

Předpokládáme lokálně rovnou zem a společný počátek kamery, těla a GNSS antény;
jejich vzájemný posun se zatím nekompenzuje. Výsledek určuje polohu na zemi,
nikoli polohu vyvýšeného objektu nad touto rovinou.

Přesnost závisí na GNSS poloze UAV, AGL, roll/pitch/yaw, kalibraci a montáži
kamery, zkreslení objektivu, určení středu objektu a skutečném tvaru terénu.
I s RTK GNSS může chyba pitch nebo výšky výrazně posunout vypočtený cíl.
Před reálným použitím zbývá dodat kalibraci, změřit montáž, připojit validní
časově sladěnou telemetrii s AGL a ověřit přesnost na známých pozemních bodech.

Referenční dokumentace: [OpenCV kalibrace a undistortPoints](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html),
[NGA WGS84](https://earth-info.nga.mil/index.php?dir=wgs84&action=wgs84),
[GNSS Toolkit – geodetické výpočty](https://gnsstk.sourceforge.net/geodesy_8c-source.html).
