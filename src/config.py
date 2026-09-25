"""Parametry příkazové řádky a jejich kontrola. Import neotevírá kameru ani serva."""
import argparse
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass(frozen=True)
class AppConfig:
    # Zdroj obrazu
    camera: int = 0
    picamera: int | None = None
    video: Path | None = None
    image: Path | None = None
    demo: bool = False
    width: int = 1296
    height: int = 972
    tuning_file: str | None = None
    ev: float | None = None
    saturation: float | None = None
    # Detekce
    redness_min: int = 25
    r_min: int = 50
    red_fraction: float = 0.35
    red_diameter_px: float | None = None
    target_diameter_m: float = 0.20
    tune: bool = False
    # Kamera a serva
    camera_calibration: Path | None = None
    hfov_deg: float = 54.0
    image_top: str = 'forward'
    gimbal: str = 'servo'                  # servo | dry-run | fixed
    fixed_angles: tuple[float, float] = (0., 0.)
    servo_x_dir: int = 1
    servo_y_dir: int = 1
    servo_calibration: Path | None = None
    gimbal_speed: float = 15.0
    scan: bool = True
    # Výsledek
    samples: int = 20
    drone_pose: tuple | None = None
    pose_file: Path | None = None
    result: Path | None = None
    once: bool = False
    output: Path | None = None
    # Běh
    headless: bool = False
    frames: int | None = None
    diagnostics: Path | None = None
    snapshot: Path | None = None


def build_parser():
    parser = argparse.ArgumentParser(
        description='Míření kamery na střed červené tečky a (volitelně) výpočet jejích souřadnic.')
    source = parser.add_argument_group('zdroj obrazu').add_mutually_exclusive_group()
    source.add_argument('--picamera', type=int, metavar='INDEX', help='CSI kamera přes Picamera2 (na dronu: 0).')
    source.add_argument('--camera', type=int, default=0, help='USB/webkamera, index (výchozí 0).')
    source.add_argument('--video', type=Path, help='Videozáznam místo kamery (serva se nehýbou).')
    source.add_argument('--image', type=Path, help='Jedna fotografie (serva se nehýbou).')
    source.add_argument('--demo', action='store_true', help='Syntetický snímek bez kamery.')
    group = parser.add_argument_group('kamera')
    group.add_argument('--width', type=int, default=1296, help='Šířka snímku (výchozí 1296 = celé zorné pole OV5647).')
    group.add_argument('--height', type=int, default=972)
    group.add_argument('--tuning-file', help='Profil libcamera (výchozí ov5647_noir.json; none = systémový).')
    group.add_argument('--ev', type=float, help='Kompenzace expozice CSI kamery, -8 až 8.')
    group.add_argument('--saturation', type=float, help='Sytost barev CSI kamery 0-32 (1 = beze změny).')
    group.add_argument('--camera-calibration', type=Path, help='JSON z calibrate_camera.py pro dané rozlišení.')
    group.add_argument('--hfov-deg', type=float, default=54.0, help='Vodorovné zorné pole bez kalibrace.')
    group.add_argument('--image-top', choices=('forward', 'right', 'backward', 'left'), default='forward',
                       help='Kam na dronu ukazuje horní okraj obrazu při kameře kolmo dolů.')
    group = parser.add_argument_group('detekce tečky')
    group.add_argument('--redness-min', type=int, default=25, help='Práh R - max(G, B).')
    group.add_argument('--r-min', type=int, default=50, help='Minimální R.')
    group.add_argument('--red-fraction', type=float, default=0.35, help='Práh (R - max(G, B)) / R.')
    group.add_argument('--red-diameter-px', type=float, help='Očekávaný průměr tečky v px (tvrdý filtr velikosti).')
    group.add_argument('--target-diameter-m', type=float, default=0.20, help='Skutečný průměr tečky (pro odhad velikosti).')
    group.add_argument('--tune', action='store_true', help='Posuvníky prahů v okně náhledu.')
    group = parser.add_argument_group('serva')
    mode = group.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Úhly počítat, ale serva nehýbat.')
    mode.add_argument('--fixed-camera', nargs=2, type=float, metavar=('RIGHT', 'FORWARD'),
                      help='Bez serv: kamera pevně v daných úhlech (např. 0 0 = kolmo dolů).')
    group.add_argument('--servo-x-dir', type=int, choices=(-1, 1), default=1, help='Otočí směr vnějšího serva.')
    group.add_argument('--servo-y-dir', type=int, choices=(-1, 1), default=1, help='Otočí směr vnitřního serva.')
    group.add_argument('--servo-calibration', type=Path, help='JSON z tools/calibrate_servos.py.')
    group.add_argument('--gimbal-speed', type=float, default=15.0, help='Max. rychlost serv ve °/s (0-60].')
    group.add_argument('--no-scan', action='store_true', help='Neprohledávat okolí, když tečka není vidět.')
    group = parser.add_argument_group('výsledek')
    group.add_argument('--samples', type=int, default=20, help='Počet vycentrovaných snímků pro medián úhlu.')
    pose = group.add_mutually_exclusive_group()
    pose.add_argument('--drone-pose', nargs=4, type=float, metavar=('LAT', 'LON', 'VYSKA_M', 'KURZ_DEG'),
                      help='Poloha dronu -> vypočítat i souřadnice tečky.')
    pose.add_argument('--pose-file', type=Path, help='JSON s polohou dronu, čte se při každém výsledku.')
    group.add_argument('--result', type=Path, help='Uložit poslední výsledek jako JSON.')
    group.add_argument('--once', action='store_true', help='Skončit po prvním výsledku.')
    group.add_argument('--output', type=Path, help='Zapisovat stav každého snímku jako JSON řádky.')
    group = parser.add_argument_group('běh')
    group.add_argument('--headless', action='store_true', help='Bez okna náhledu (přes SSH).')
    group.add_argument('--frames', type=int, help='Skončit po daném počtu snímků.')
    group.add_argument('--diagnostics', type=Path, help='Každé 2 s uložit snímek + stav (max. 30) pro ladění.')
    group.add_argument('--snapshot', type=Path, help='Jen uložit jednu fotku z kamery a skončit.')
    return parser


def parse_args(argv=None) -> AppConfig:
    parser = build_parser()
    a = parser.parse_args(argv)
    live = not (a.video or a.image or a.demo)
    if a.fixed_camera is not None:
        if not all(math.isfinite(v) and abs(v) <= 90 for v in a.fixed_camera):
            parser.error('--fixed-camera: úhly v rozsahu -90 až 90°.')
        gimbal, fixed = 'fixed', tuple(a.fixed_camera)
    elif not live:
        gimbal, fixed = 'fixed', (0., 0.)     # záznam/fotka nereaguje na serva: kamera jako pevná
    elif a.dry_run:
        gimbal, fixed = 'dry-run', (0., 0.)
    else:
        gimbal, fixed = 'servo', (0., 0.)
    if a.width < 1 or a.height < 1:
        parser.error('Rozlišení musí být kladné.')
    if not 0 < a.gimbal_speed <= 60 or not math.isfinite(a.gimbal_speed):
        parser.error('--gimbal-speed musí být v rozsahu (0, 60].')
    if a.samples < 3:
        parser.error('--samples musí být alespoň 3.')
    if not 1 < a.hfov_deg < 179:
        parser.error('--hfov-deg musí být mezi 1 a 179°.')
    if not (0 <= a.redness_min <= 255 and 0 <= a.r_min <= 255 and 0 <= a.red_fraction <= 1):
        parser.error('Prahy červené: --redness-min a --r-min 0-255, --red-fraction 0-1.')
    if a.red_diameter_px is not None and not (math.isfinite(a.red_diameter_px) and a.red_diameter_px > 0):
        parser.error('--red-diameter-px musí být kladné.')
    if not (math.isfinite(a.target_diameter_m) and a.target_diameter_m > 0):
        parser.error('--target-diameter-m musí být kladné.')
    if (a.tuning_file or a.ev is not None or a.saturation is not None) and a.picamera is None:
        parser.error('--tuning-file, --ev a --saturation jsou jen pro --picamera.')
    if a.ev is not None and not -8 <= a.ev <= 8:
        parser.error('--ev musí být v rozsahu -8 až 8.')
    if a.saturation is not None and not 0 <= a.saturation <= 32:
        parser.error('--saturation musí být v rozsahu 0 až 32.')
    if a.tune and a.headless:
        parser.error('--tune potřebuje okno (bez --headless).')
    if a.frames is not None and a.frames < 1:
        parser.error('--frames musí být kladné.')
    if a.snapshot and (a.video or a.image or a.demo):
        parser.error('--snapshot je jen pro živou kameru.')
    if a.drone_pose is not None:
        from .locate import DronePosition
        try:
            DronePosition(*a.drone_pose)
        except ValueError as error:
            parser.error(str(error))
    values = {k: v for k, v in vars(a).items() if k not in ('dry_run', 'fixed_camera', 'no_scan')}
    values.update(gimbal=gimbal, fixed_angles=fixed, scan=not a.no_scan,
                  drone_pose=tuple(a.drone_pose) if a.drone_pose else None)
    return AppConfig(**values)
