"""Hledání po řádcích a přechod na čerstvé měření cíle. Bez letových rozhraní.

Body trasy se ořezávají podle zakázaných čar z pásky: dron doletí jen k pásce
(s rezervou) a pokračuje dalším řádkem. Cíl za páskou se nepřijme.
"""
import math
from dataclasses import replace
from .approach import ApproachController, TargetEstimate
from .field import GroundPoint


def clip_to_tape(start, end, field, margin):
    """Nejvzdálenější bod úsečky start->end s odstupem od pásky >= margin.

    Pokud start rezervu nemá (páska se objevila pozdě), vrátí bod odsunutý od
    pásky dovnitř. Bez pásky vrací end beze změny.
    """
    if field is None or not field.forbidden_lines:
        return end
    close = [line for line in field.forbidden_lines if line.signed_distance(start) < margin]
    if close:
        n, e = start.north_m, start.east_m
        for line in close:
            push = margin+.2-line.signed_distance(start)
            dn, de = line.inward()
            n, e = n+dn*push, e+de*push
        return GroundPoint(n, e)
    t_max = 1.
    for line in field.forbidden_lines:
        d0, d1 = line.signed_distance(start), line.signed_distance(end)
        if d1 < margin:
            t_max = min(t_max, (d0-margin)/(d0-d1))
    return GroundPoint(start.north_m+(end.north_m-start.north_m)*t_max,
                       start.east_m+(end.east_m-start.east_m)*t_max)


class SearchMission:
    def __init__(self, sweep, *, tape_margin=1.):
        self.sweep = sweep
        self.tape_margin = tape_margin
        self.controller = ApproachController(bounds=sweep.bounds, dwell=.3)
        self.index = 0
        self.tracking = False
        self.last_seen = None

    def update(self, vehicle, target, now, field=None):
        usable = (target is not None and target.valid and
                  all(math.isfinite(v) for v in (now, target.sampled_at, target.error_m,
                      target.position.north_m, target.position.east_m)) and
                  0 <= now-target.sampled_at <= .25 and 0 <= target.error_m <= 2 and
                  self.controller.inside(target.position, target.error_m+self.controller.tolerance) and
                  (field is None or field.line_clearance(target.position) >= .5))
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
        point = clip_to_tape(vehicle.position, self.sweep.points[self.index], field, self.tape_margin)
        waypoint = TargetEstimate(point, now, f'waypoint:{self.index}')
        self.controller.dwell = .3
        command = self.controller.update(vehicle, waypoint, now)
        if command.state == 'ARRIVED':
            self.index += 1
        if command.state in ('ACQUIRING_TARGET', 'APPROACHING', 'SETTLING', 'ARRIVED'):
            command = replace(command, state='SEARCHING')
        return command
