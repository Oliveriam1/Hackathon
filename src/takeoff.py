"""Potvrzení výšky při simulačním vzletu; bez přístupu k hardwaru."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TakeoffResult:
    state: str
    up_m_s: float = 0.
    ready: bool = False


class TakeoffController:
    def __init__(self, height=5.):
        if not math.isfinite(height) or not 1 <= height <= 20:
            raise ValueError('Simulační výška musí být 1 až 20 m.')
        self.height = height
        self.started = self.last_time = self.settled = None
        self.failure = None

    def update(self, height, up_speed, sampled_at, now):
        if self.failure:
            return TakeoffResult(self.failure)
        if not all(math.isfinite(x) for x in (height, up_speed, sampled_at, now)):
            self.failure = 'TAKEOFF_INVALID_DATA'
        elif not 0 <= now-sampled_at <= .3:
            self.failure = 'TAKEOFF_STALE'
        elif self.last_time is not None and not 0 < now-self.last_time <= .3:
            self.failure = 'TAKEOFF_STREAM_GAP'
        if self.failure:
            return TakeoffResult(self.failure)
        if self.started is None:
            self.started = now
        self.last_time = now
        if now-self.started > 60:
            self.failure = 'TAKEOFF_TIMEOUT'
            return TakeoffResult(self.failure)
        error = self.height-height
        # Simulátor používá kladnou rychlost nahoru, nikoli NED vz.
        speed = max(-.5, min(1., error*.8))
        if abs(error) <= .2 and abs(up_speed) <= .15:
            if self.settled is None:
                self.settled = now
            if now-self.settled >= 1.:
                return TakeoffResult('AIRBORNE', speed, True)
            return TakeoffResult('TAKEOFF_SETTLING', speed)
        self.settled = None
        return TakeoffResult('TAKING_OFF', speed)
