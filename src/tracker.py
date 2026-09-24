"""Navazování detekcí mezi snímky: potvrzení, vyhlazení a překlenutí výpadků.

Nový cíl vzniká jen z jediného kandidáta "kolečko v obdélníku". Potvrzený
cíl přečká krátký výpadek (predikce polohy) a omezenou dobu jej může držet
i samotné kolečko na očekávaném místě, např. při přeexponovaném okraji listu.
"""
from dataclasses import dataclass, replace
import math

CONFIRM_HITS = 4       # shodné detekce pro potvrzení cíle
MAX_TENTATIVE_MISSED = 1
MAX_MISSED = 10        # snímky bez měření, po které potvrzený cíl ještě držíme
MAX_CIRCLE_ONLY = 15   # snímky, po které potvrzený cíl drží jen kolečko bez obdélníku
POSITION_GAIN = 0.7    # alfa-beta filtr: vyšší = rychlejší reakce, méně vyhlazení
VELOCITY_GAIN = 0.3
SIZE_GAIN = 0.5


@dataclass(frozen=True)
class TrackState:
    target: object | None  # vyhlazená poloha (Circle) potvrzeného cíle
    confirmed: bool
    measured: bool         # cíl byl v tomto snímku skutečně změřen
    status: str


class TargetTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.track = None      # vyhlazený Circle
        self.velocity = (0.0, 0.0)
        self.hits = 0
        self.missed = 0
        self.circle_only = 0
        self.confirmed = False

    @property
    def predicted(self):
        if self.track is None:
            return None
        return self.track.shifted(*self.velocity)

    def _match(self, circles, size_tolerance):
        prediction = self.predicted
        base = max(25.0, 1.5*prediction.radius)
        # Brána roste během výpadku, ale omezeně: nepřeskočit na jiný terč.
        gate = base + min(10*self.missed, base)
        best, best_score = None, math.inf
        for circle in circles:
            # Chybný odhad rychlosti nesmí cíl ztratit: platí i poslední poloha.
            distance = min(math.hypot(circle.x-prediction.x, circle.y-prediction.y),
                           math.hypot(circle.x-self.track.x, circle.y-self.track.y))
            size = abs(math.log(circle.radius/prediction.radius)) / math.log(size_tolerance)
            if distance >= gate or size >= 1:
                continue
            # Blízko a stejně velké; jinak by vyhrál i větší kruh opřený o rám.
            score = distance/gate + size
            if score < best_score:
                best, best_score = circle, score
        return best

    def update(self, candidates, search_near=None):
        """candidates: kolečka v obdélníku; search_near(predikce) -> samotná kolečka."""
        if self.track is None:
            return self._start(candidates)
        match = self._match(candidates, 1.6)
        circle_only = False
        if match is None and self.confirmed and search_near and self.circle_only < MAX_CIRCLE_ONLY:
            match = self._match(search_near(self.predicted), 1.3)
            circle_only = match is not None
        if match is None:
            return self._miss(candidates)
        self._correct(match)
        self.hits += not circle_only
        self.missed = 0
        self.circle_only = self.circle_only + 1 if circle_only else 0
        self.confirmed = self.confirmed or self.hits >= CONFIRM_HITS
        if not self.confirmed:
            return TrackState(None, False, True, f'KANDIDAT {self.hits}/{CONFIRM_HITS}')
        status = f'TERC POTVRZEN - BEZ OBDELNIKU {self.circle_only}/{MAX_CIRCLE_ONLY}' if circle_only else 'TERC POTVRZEN'
        return TrackState(self.track, True, True, status)

    def _start(self, candidates):
        if len(candidates) != 1:
            return TrackState(None, False, False, 'VICE KANDIDATU' if candidates else 'HLEDAM KOLECKO VE CTYRUHELNIKU')
        self.track, self.hits = candidates[0], 1
        return TrackState(None, False, True, f'KANDIDAT 1/{CONFIRM_HITS}')

    def _correct(self, circle):
        prediction = self.predicted
        dx, dy = circle.x-prediction.x, circle.y-prediction.y
        vx, vy = self.velocity[0] + VELOCITY_GAIN*dx, self.velocity[1] + VELOCITY_GAIN*dy
        # Jeden odlehlý snímek nesmí rozjet predikci mimo terč.
        limit = max(15.0, prediction.radius)
        speed = math.hypot(vx, vy)
        self.velocity = (vx, vy) if speed <= limit else (vx*limit/speed, vy*limit/speed)
        radius = prediction.radius + SIZE_GAIN*(circle.radius-prediction.radius)
        self.track = replace(circle, x=prediction.x + POSITION_GAIN*dx, y=prediction.y + POSITION_GAIN*dy,
                             radius=radius, minor=circle.minor_radius*radius/circle.radius)

    def _miss(self, candidates):
        self.missed += 1
        if self.missed > (MAX_MISSED if self.confirmed else MAX_TENTATIVE_MISSED):
            self.reset()
            return self._start(candidates)
        # Predikce s tlumením rychlosti, aby odhad během výpadku neujel.
        self.track = self.predicted
        self.velocity = (self.velocity[0]*0.7, self.velocity[1]*0.7)
        if not self.confirmed:
            return TrackState(None, False, False, f'KANDIDAT {self.hits}/{CONFIRM_HITS}')
        return TrackState(self.track, True, False, f'TERC POTVRZEN - PREDIKCE {self.missed}/{MAX_MISSED}')
