"""Simulační hledání a přechod na čerstvé měření cíle. Bez letových rozhraní."""
import math
from dataclasses import replace
from .approach import ApproachController, TargetEstimate


class SearchMission:
    def __init__(self, sweep):
        self.sweep = sweep
        self.controller = ApproachController(bounds=sweep.bounds, dwell=.3)
        self.index = 0
        self.tracking = False
        self.last_seen = None

    def update(self, vehicle, target, now):
        usable = (target is not None and target.valid and
                  all(math.isfinite(v) for v in (now, target.sampled_at, target.error_m,
                      target.position.north_m, target.position.east_m)) and
                  0 <= now-target.sampled_at <= .25 and 0 <= target.error_m <= 2 and
                  self.controller.inside(target.position, target.error_m+self.controller.tolerance))
        if usable:
            self.tracking = True
            self.last_seen = now
        if self.tracking:
            if usable:
                # Oddělené identity: bod trasy nemůže být zaměněn za detekci.
                self.controller.dwell = 1.
                return self.controller.update(vehicle, replace(target, identity='target:'+target.identity), now)
            if not math.isfinite(now) or now-self.last_seen < 2:
                command = self.controller.update(vehicle, None, now)
                return replace(command, state='TARGET_LOST_HOLD' if command.state == 'TARGET_LOST' else command.state)
            self.tracking = False
        if self.index >= len(self.sweep.points):
            command = self.controller.update(vehicle, None, now)
            return replace(command, state='SEARCH_COMPLETE' if command.state == 'TARGET_LOST' else command.state)
        waypoint = TargetEstimate(self.sweep.points[self.index], now, f'waypoint:{self.index}')
        self.controller.dwell = .3
        command = self.controller.update(vehicle, waypoint, now)
        if command.state == 'ARRIVED':
            self.index += 1
        if command.state in ('ACQUIRING_TARGET', 'APPROACHING', 'SETTLING', 'ARRIVED'):
            command = replace(command, state='SEARCHING')
        return command
