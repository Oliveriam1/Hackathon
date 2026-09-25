"""Navazování detekcí červené tečky mezi snímky: potvrzení, vyhlazení, výpadky.

Nový cíl se potvrdí až po několika shodných měřeních v čase. Predikce polohy
nikdy není nové měření. Nejednoznačné asociace neaktualizují cíl.
"""
from dataclasses import dataclass, replace
import math
import time


@dataclass(frozen=True)
class TrackState:
    target: object | None  # vyhlazená poloha (Circle) potvrzeného cíle
    confirmed: bool
    measured: bool         # cíl byl v tomto snímku skutečně změřen
    status: str


class TimedTargetTracker:
    """Jediný vlastník identity a ROI červeného cíle; rychlost v px/s.

    Predikce nikdy není nové měření. Nejednoznačné asociace neaktualizují cíl.
    """
    def __init__(self, confirm_s=0.25, lost_s=0.7):
        self.confirm_s, self.lost_s = confirm_s, lost_s
        self.next_id = 1
        self.reset()

    def reset(self):
        self.track = self.last_measurement = None
        self.velocity = (0., 0.)
        self.last_time = self.last_seen = self.first_seen = None
        self.confirmed = False
        self.hits = 0
        self.track_id = None
        self.status = 'SEARCHING'
        self.last_full_scan = None

    def prediction(self, now):
        if self.track is None:
            return None
        dt = max(0., min(now-self.last_seen, self.lost_s))
        return self.track.shifted(self.velocity[0]*dt, self.velocity[1]*dt)

    def search_roi(self, shape, now):
        if (not self.confirmed or self.last_seen is None or now-self.last_seen > 0.15 or
                self.last_full_scan is None or now-self.last_full_scan >= 0.5):
            self.last_full_scan = now
            return None
        c = self.prediction(now)
        age = max(0., now-self.last_seen)
        reach = max(60., c.radius*5) + math.hypot(*self.velocity)*age + 100*age
        height, width = shape[:2]
        roi = (max(0, int(c.x-reach)), max(0, int(c.y-reach)),
               min(width, int(c.x+reach)+1), min(height, int(c.y+reach)+1))
        return roi if roi[2]-roi[0] > 4 and roi[3]-roi[1] > 4 else None

    def update(self, candidates, *, sample_time=None, scores=None):
        now = time.monotonic() if sample_time is None else sample_time
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            self.last_measurement = None
            self.status = 'STALE_OR_REPEATED'
            return TrackState(self.track, self.confirmed, False, self.status)
        scores = scores if scores is not None else [0.8]*len(candidates)
        if self.last_seen is not None and now-self.last_seen > (self.lost_s if self.confirmed else 0.3):
            self.reset()
        self.last_time = now
        self.last_measurement = None
        chosen = None
        ambiguous = False
        if self.track is None:
            ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
            ambiguous = len(ranked) > 1 and ranked[0][0]-ranked[1][0] < 0.18
            if ranked and not ambiguous:
                chosen = ranked[0][1]
        else:
            predicted = self.prediction(now)
            age = now-self.last_seen
            gate = max(20., self.track.radius*2) + min(100., 120*age)
            ranked = []
            for circle, quality in zip(candidates, scores):
                distance = math.hypot(circle.x-predicted.x, circle.y-predicted.y)
                size = abs(math.log(circle.radius/self.track.radius))
                if distance < gate and size < math.log(1.8):
                    ranked.append((distance/gate + size*0.4 - quality*0.2, circle))
            ranked.sort(key=lambda item: item[0])
            ambiguous = len(ranked) > 1 and ranked[1][0]-ranked[0][0] < 0.20
            if ranked and not ambiguous:
                chosen = ranked[0][1]
        if chosen is None:
            self.status = 'AMBIGUOUS' if ambiguous else ('PREDICTION' if self.confirmed else 'SEARCHING')
            if not self.confirmed:
                self.first_seen, self.hits = None, 0
            return TrackState(self.prediction(now) if self.confirmed else None,
                              self.confirmed, False, self.status)
        if self.track is None:
            self.track = chosen
            self.track_id, self.next_id = self.next_id, self.next_id+1
            self.first_seen, self.hits = now, 1
        else:
            dt = max(1e-3, now-self.last_seen)
            if dt > 0.25 and not self.confirmed:
                self.first_seen, self.hits = now, 0
            predicted = self.prediction(now)
            dx, dy = chosen.x-predicted.x, chosen.y-predicted.y
            alpha = 1-math.exp(-dt/0.06)
            beta = min(0.5, 0.25*alpha)
            vx, vy = self.velocity[0]+beta*dx/dt, self.velocity[1]+beta*dy/dt
            speed = math.hypot(vx, vy)
            self.velocity = (vx, vy) if speed <= 1500 else (vx*1500/speed, vy*1500/speed)
            self.track = replace(chosen, x=predicted.x+alpha*dx, y=predicted.y+alpha*dy)
            self.first_seen = now if self.first_seen is None else self.first_seen
            self.hits += 1
        self.last_seen = now
        self.last_measurement = chosen
        self.confirmed = self.confirmed or (self.hits >= 4 and now-self.first_seen >= self.confirm_s)
        self.status = 'TRACKING' if self.confirmed else 'CANDIDATE'
        return TrackState(self.track if self.confirmed else None, self.confirmed, True, self.status)

    def diagnostics(self):
        return dict(state=self.status, track_id=self.track_id, hits=self.hits,
                    velocity_px_s=list(self.velocity),
                    measurement_age_s=None if self.last_seen is None else self.last_time-self.last_seen)
