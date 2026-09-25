"""Míření kamery na střed červené tečky – hlavní výstup programu.

Výstupem jsou dva úhly kamery vůči tělu dronu (GimbalAngles):
  right   – náklon doprava (+) / doleva (-), stupně
  forward – náklon dopředu (+) / dozadu (-), stupně
  0 / 0   – kamera míří kolmo dolů.

Úhel „na střed tečky“ se nepočítá jen z polohy serv. Serva mají necitlivost
(deadband) a tečka nemusí být přesně ve středu obrazu, proto se ke skutečnému
natočení kamery přičte zbývající odchylka pixelu přes model objektivu
(geometry.centering_angles). Výsledek je úhel, při kterém by střed tečky ležel
přesně v optické ose, i když serva stojí o kousek vedle.

Stavy:
  SEARCHING  – tečka není v obraze (nebo ještě není potvrzená)
  SCANNING   – tečka dlouho není vidět, serva prohledávají okolí
  TRACKING   – tečka potvrzená, serva ji dotahují do středu
  CENTERED   – tečka je stabilně ve středu obrazu (≥ 0.5 s), úhel je platný
  LOST       – potvrzená tečka zmizela, drží se poslední poloha kamery
  AMBIGUOUS  – více podobných teček, nic se neměří
"""
import math
import statistics
import time
from dataclasses import dataclass
from .geometry import GimbalAngles, centering_angles
from .gimbal_controller import GimbalController, GimbalScanner
from .target_lock import TargetLock


@dataclass(frozen=True)
class AimState:
    state: str
    camera_angles: GimbalAngles           # kam kamera míří při tomto snímku
    target_angles: GimbalAngles | None    # úhly na střed tečky (jen s čerstvým měřením)
    centered: bool                        # tečka stabilně ve středu obrazu
    pixel_error: tuple[float, float] | None  # odchylka středu tečky od středu obrazu, px
    sample_time: float
    track_id: int | None = None


@dataclass(frozen=True)
class AimResult:
    """Zprůměrovaný (medián) úhel z několika vycentrovaných snímků."""
    right: float
    forward: float
    samples: int
    spread_deg: float        # medián odchylky jednotlivých měření od výsledku
    track_id: int | None
    measured_at: float       # time.monotonic() posledního měření

    @property
    def angles(self):
        return GimbalAngles(self.right, self.forward)


class Aimer:
    """Jeden krok míření pro každý snímek. Serva ovládá, pokud dostane `gimbal`.

    gimbal=None: pevná kamera (serva se nehýbou, úhly = fixed_angles),
    dry_run=True: úhly se počítají, ale na serva se nic neposílá.
    """
    def __init__(self, camera_model, *, gimbal=None, image_top='forward', max_speed=15.,
                 scan=True, fixed_angles=GimbalAngles(), dry_run=False, scan_after_s=1.5):
        self.model, self.gimbal, self.image_top = camera_model, gimbal, image_top
        self.dry_run, self.fixed_angles = dry_run, fixed_angles
        self.movable = gimbal is not None or dry_run
        self.controller = GimbalController(camera_model, image_top=image_top, max_speed=max_speed)
        self.scanner = GimbalScanner(max_speed=min(20., max_speed)) if scan and self.movable else None
        self.scan_after_s = scan_after_s
        self.lock = TargetLock()
        self.angles = fixed_angles
        self.seen_at = None

    def current_angles(self):
        if self.gimbal is not None and not self.dry_run:
            return GimbalAngles(self.gimbal.x, self.gimbal.y)
        return self.angles

    def _move(self, angles):
        if self.gimbal is not None and not self.dry_run:
            self.gimbal.move_to(angles.right, angles.forward)
            angles = GimbalAngles(self.gimbal.x, self.gimbal.y)  # po omezení rozsahu
        self.angles = angles

    def update(self, observation, *, sample_time, now=None):
        now = time.monotonic() if now is None else now
        size = tuple(observation.frame_size)
        if tuple(self.model.size) != size:
            raise ValueError(f'Model kamery je pro {tuple(self.model.size)}, snímek má {size}.')
        current = self.current_angles()  # natočení platné pro tento snímek
        if self.seen_at is None:
            self.seen_at = now  # prohledávat až po scan_after_s bez tečky, ne hned po startu
        lock = self.lock.update(observation, sample_time=sample_time, now=now)
        target_angles = pixel_error = None
        point = observation.measurement
        if lock['locked'] and point is not None:
            target_angles = centering_angles((point.x, point.y), current, self.model, self.image_top)
            pixel_error = (point.x-size[0]/2, point.y-size[1]/2)
        state = {'LOCKED': 'TRACKING', 'IMAGE_CENTERED': 'CENTERED'}.get(lock['state'], lock['state'])
        if state == 'SEARCHING' and observation.confirmed:
            state = 'LOST'
        if state == 'STALE_OR_REPEATED':
            state = 'STALE'
        if self.movable:
            command = self.controller.update(observation, current, sample_time=sample_time, now=now)
            if observation.confirmed:
                self.seen_at = now
                if self.scanner is not None:
                    self.scanner.reset()
            elif (self.scanner is not None and command is None and state in ('SEARCHING', 'LOST')
                  and (self.seen_at is None or now-self.seen_at > self.scan_after_s)):
                command = self.scanner.update(current, now)
                state = 'SCANNING'
            if command is not None:
                self._move(command)
        tracking = (observation.red or {}).get('tracking', {})
        return AimState(state, current, target_angles, lock['image_centered'], pixel_error,
                        sample_time, tracking.get('track_id'))


class AngleAverager:
    """Sbírá úhly z po sobě jdoucích snímků téže tečky a vrací medián.

    require_centered=True (serva): jen snímky, kdy je tečka stabilně ve středu
    obrazu – nejpřesnější, zkreslení objektivu se neuplatní.
    require_centered=False (pevná kamera): stačí potvrzené čerstvé měření.
    """
    def __init__(self, samples=20, max_gap_s=1., require_centered=True):
        if samples < 3:
            raise ValueError('Pro medián jsou potřeba alespoň 3 vzorky.')
        self.samples, self.max_gap_s, self.require_centered = samples, max_gap_s, require_centered
        self.values = []
        self.track_id = None
        self.last_time = None

    def reset(self):
        self.values, self.track_id, self.last_time = [], None, None

    def add(self, aim):
        """Přidá vzorek; při jiné tečce nebo dlouhé mezeře začne znovu."""
        if aim.target_angles is None or (self.require_centered and not aim.centered):
            if aim.state not in ('TRACKING', 'CENTERED'):
                self.reset()  # tečka ztracena: nemíchat měření z různých okamžiků
            return
        if self.last_time is not None and aim.sample_time <= self.last_time:
            return  # opakovaný snímek
        if (aim.track_id != self.track_id or
                (self.last_time is not None and aim.sample_time-self.last_time > self.max_gap_s)):
            self.values = []
        self.track_id, self.last_time = aim.track_id, aim.sample_time
        self.values.append((aim.target_angles.right, aim.target_angles.forward))
        self.values = self.values[-self.samples:]

    def result(self):
        if len(self.values) < self.samples:
            return None
        right = statistics.median(v[0] for v in self.values)
        forward = statistics.median(v[1] for v in self.values)
        spread = statistics.median(math.hypot(v[0]-right, v[1]-forward) for v in self.values)
        return AimResult(right, forward, len(self.values), spread, self.track_id, self.last_time)
