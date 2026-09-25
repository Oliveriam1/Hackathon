"""Konfigurace aplikace a validace CLI; import neotevírá hardware."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Dosavadní nastavení hardwaru; změna struktury nemění pulzy ani zapojení.
X_PIN, Y_PIN = 18, 13
X_CENTER_US, Y_CENTER_US = 1500, 1500
US_PER_DEG = 1000.0 / 90.0
X_DIR, Y_DIR = 1, 1
X_LIMITS = (-60.0, 60.0)
Y_LIMITS = (-45.0, 45.0)


@dataclass(frozen=True)
class StartReference:
    """Budoucí počátek lokální mapy; WGS84, nikoli detekce zelené barvy."""
    latitude_deg: float
    longitude_deg: float

    def __post_init__(self):
        if not (-90 <= self.latitude_deg <= 90 and -180 <= self.longitude_deg <= 180):
            raise ValueError('Neplatné souřadnice startovní reference.')


@dataclass(frozen=True)
class MissionConfig:
    """Rezervovaná konfigurace mise; zatím se nenačítá z CLI ani neřídí let.

    Neznámé hodnoty zůstávají None. AppConfig.altitude není výška vzletu.
    """
    start: StartReference | None = None
    takeoff_height_m: float | None = None


@dataclass(frozen=True)
class AppConfig:
    """Ověřené CLI hodnoty. altitude je odhad pro detekci, nikoli povel ke vzletu."""
    camera: int = 0
    image: Path | None = None
    video: Path | None = None
    demo: bool = False
    picamera: int | None = None
    snapshot: Path | None = None
    headless: bool = False
    status: bool = False
    drone_data: bool = False
    diagnostics: Path | None = None
    record_dir: Path | None = None
    sensitivity: float = 1.5
    hough: bool = False
    detector: str = 'red'
    saturation: float | None = None
    redness_min: int = 25
    red_fraction: float = 0.35
    r_min: int = 50
    tune: bool = False
    altitude: float | None = None
    target_diameter_m: float = 0.20
    hfov_deg: float = 54.0
    camera_calibration: Path | None = None
    geometry_debug: bool = False
    red_diameter_px: float | None = None
    tuning_file: str | None = None
    output: Path | None = None
    frames: int | None = None
    width: int = 640
    height: int = 480
    mavlink: str | None = None
    baud: int = 115200
    target_system: int | None = None
    ev: float | None = None
    servo: bool = False
    jako_red_tracker: bool = False


def parse_args(argv=None) -> AppConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Index kamery (výchozí: 0).")
    source.add_argument("--image", type=Path, help="Obrázek místo živé kamery.")
    source.add_argument("--video", type=Path, help="Videozáznam místo živé kamery, např. z letu.")
    source.add_argument("--demo", action="store_true", help="Testovací obraz bez kamery.")
    source.add_argument("--picamera", type=int, metavar="INDEX", help="CSI kamera přes Picamera2, např. --picamera 0.")
    parser.add_argument("--snapshot", type=Path, help="Uloží jeden snímek bez grafického okna (např. test.jpg).")
    parser.add_argument('--headless', action='store_true', help='Jen data, bez grafického okna; ukončení Ctrl+C.')
    parser.add_argument('--status', action='store_true', help='Čitelný stav 1x za sekundu místo JSON v terminálu.')
    parser.add_argument('--drone-data', action='store_true', help='Pouze JSON kontrakt vizuálních dat pro řídicí část, bez odesílání povelů.')
    parser.add_argument('--diagnostics', type=Path, help='Uloží nejvýše 30 dvojic raw/marked snímků, každé 2 s.')
    parser.add_argument('--record-dir', type=Path, help='Uloží každý zpracovaný snímek a čas pro offline vyhodnocení; délku omezte --frames.')
    parser.add_argument('--sensitivity', type=float, choices=(1, 1.5, 2, 3), default=1.5)
    parser.add_argument('--hough', action='store_true', help='Pomalá záloha pro přerušené kruhové hrany (pro porovnání).')
    parser.add_argument('--detector', choices=('geometry', 'red'), default='red',
                        help='red (výchozí) = červené/růžové tečky s kontrolou tvaru a trackingem; geometry = kolečko v obdélníku.')
    parser.add_argument('--saturation', type=float,
                        help='Sytost barev CSI kamery 0-32 (libcamera Saturation, 1 = beze změny). '
                             'Výchozí nastavení kamery jako red_tracker.py; např. 2 zvýrazní červenou.')
    parser.add_argument('--redness-min', type=int, default=25, help='Práh R - max(G, B), výchozí 25 jako red_tracker.py.')
    parser.add_argument('--red-fraction', type=float, default=0.35,
                        help='Práh (R - max(G, B)) / R, výchozí 0.35; nižší přijme i matnou červenou.')
    parser.add_argument('--r-min', type=int, default=50, help='Minimální R, výchozí 50.')
    parser.add_argument('--tune', action='store_true', help='Posuvníky prahů červené a sytosti v okně (jako red_tracker --tune).')
    parser.add_argument('--altitude', type=float, help='Ruční odhad výšky v metrech; pouze měkké skóre velikosti.')
    parser.add_argument('--target-diameter-m', type=float, default=0.20, help='Skutečný průměr terče v metrech.')
    parser.add_argument('--hfov-deg', type=float, default=54.0, help='Jmenovitý horizontální zorný úhel.')
    parser.add_argument('--camera-calibration', type=Path, help='JSON kalibrace CameraModel pro aktuální rozlišení.')
    parser.add_argument('--geometry-debug', action='store_true', help='Volitelný geometrický náhled okolí kandidáta.')
    parser.add_argument('--red-diameter-px', type=float, help='Očekávaný průměr červené tečky v pixelech; jinak bez filtru velikosti.')
    parser.add_argument('--tuning-file', help='CSI profil (výchozí ov5647_noir.json: kamera bez IR filtru, jinak růžový obraz). '
                                              'Hodnota none ponechá systémový profil.')
    parser.add_argument('--output', type=Path, help='Připojuje JSON Lines do souboru; jinak zapisuje na stdout.')
    parser.add_argument('--frames', type=int, help='Ukončit po daném počtu nových snímků.')
    parser.add_argument('--width', type=int, default=640, help='Šířka CSI snímku.')
    parser.add_argument('--height', type=int, default=480, help='Výška CSI snímku.')
    parser.add_argument('--mavlink', help='Port/endpoint ArduPilotu; jen telemetrie, např. /dev/ttyACM0.')
    parser.add_argument('--baud', type=int, default=115200, help='Rychlost sériového MAVLink spojení.')
    parser.add_argument('--target-system', type=int, help='Očekávané MAVLink system ID autopilota.')
    parser.add_argument('--ev', type=float, default=None,
                        help='Kompenzace expozice CSI kamery, např. --ev -2. Jinak výchozí nastavení profilu.')
    parser.add_argument('--servo', action='store_true',
                        help='Serva závěsu přes pigpio (BCM 18 a 13) jako red_tracker.py; najedou do 0/0.')
    parser.add_argument('--jako-red-tracker', action='store_true',
                        help='Vše jako red_tracker.py: CSI 1296x972, profil ov5647_noir.json, serva --servo.')
    args = parser.parse_args(argv)
    if args.drone_data and (args.status or args.snapshot):
        parser.error('--drone-data nelze kombinovat s --status ani --snapshot.')
    if args.jako_red_tracker:
        if args.image or args.video or args.demo:
            parser.error('--jako-red-tracker vyžaduje CSI kameru.')
        args.picamera = 0 if args.picamera is None else args.picamera
        args.width, args.height = 1296, 972  # plné zorné pole OV5647, binning 2x2
        args.tuning_file = args.tuning_file or 'ov5647_noir.json'
        args.servo = True
    if args.red_diameter_px is not None and (not np.isfinite(args.red_diameter_px) or args.red_diameter_px <= 0 or args.detector != 'red'):
        parser.error('--red-diameter-px musí být kladné číslo a vyžaduje --detector red.')
    if args.tuning_file is not None and args.picamera is None:
        parser.error('--tuning-file vyžaduje --picamera.')
    if args.hough and args.detector == 'red':
        parser.error('--hough je pouze pro --detector geometry.')
    if args.saturation is not None and (not np.isfinite(args.saturation) or not 0 <= args.saturation <= 32):
        parser.error('--saturation musí být v rozsahu 0 až 32.')
    if not (0 <= args.redness_min <= 255 and 0 <= args.r_min <= 255 and 0 <= args.red_fraction <= 1):
        parser.error('Prahy červené: --redness-min a --r-min 0-255, --red-fraction 0-1.')
    if args.tune and (args.headless or args.detector != 'red'):
        parser.error('--tune vyžaduje okno (bez --headless) a --detector red.')
    if args.altitude is not None and (not np.isfinite(args.altitude) or args.altitude <= 0):
        parser.error('--altitude musí být kladná výška v metrech.')
    if not np.isfinite(args.target_diameter_m) or args.target_diameter_m <= 0:
        parser.error('--target-diameter-m musí být kladné číslo.')
    if not np.isfinite(args.hfov_deg) or not 1 < args.hfov_deg < 179:
        parser.error('--hfov-deg musí být mezi 1 a 179 stupni.')
    if args.frames is not None and args.frames < 1:
        parser.error('--frames musí být kladné.')
    if args.baud <= 0 or (args.target_system is not None and not 1 <= args.target_system <= 255):
        parser.error('Neplatné --baud nebo --target-system.')
    if args.mavlink and (args.demo or args.image or args.video or args.snapshot):
        parser.error('--mavlink připojujte pouze k živé detekci, nikoliv k záznamu nebo fotografii.')
    if args.width < 1 or args.height < 1:
        parser.error('Rozlišení musí být kladné.')
    if args.output and any(path and path.resolve() == args.output.resolve() for path in (args.image, args.video)):
        parser.error('Výstupní data nesmí přepisovat vstupní obraz/video.')
    if args.snapshot and (args.output or args.headless or args.frames or args.status or args.diagnostics or args.record_dir):
        parser.error('--snapshot je samostatný režim fotografie; pro data použijte --headless.')
    if args.camera < 0:
        parser.error("Index kamery musí být nezáporný.")
    if args.picamera is not None and args.picamera < 0:
        parser.error("Index CSI kamery musí být nezáporný.")
    if args.ev is not None and args.picamera is None:
        parser.error('--ev lze použít pouze s --picamera.')
    if args.ev is not None and (not np.isfinite(args.ev) or not -8 <= args.ev <= 8):
        parser.error('--ev musí být v rozsahu -8 až 8.')
    return AppConfig(**vars(args))

