"""Konzervativní kontrola ověřené konvexní mapy a pásky v metrech. Bez letových povelů."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneDecision:
    allowed: bool
    state: str
    clearance_m: float | None = None
    required_clearance_m: float | None = None


class ZoneGuard:
    def __init__(self, *, margin=.5, acceleration=1., reaction_time=.3, max_map_age=2.):
        if not all(math.isfinite(v) and v > 0 for v in (margin, acceleration, reaction_time, max_map_age)):
            raise ValueError('Neplatné parametry zóny.')
        self.margin, self.acceleration = margin, acceleration
        self.reaction_time, self.max_map_age = reaction_time, max_map_age

    def check(self, field, vehicle, command, now):
        points = field.verified_boundary
        if len(points) < 3 or field.observed_at is None:
            return ZoneDecision(False, 'ZONE_UNKNOWN')
        values = (now, field.observed_at, vehicle.sampled_at, vehicle.position.north_m,
                  vehicle.position.east_m, vehicle.velocity_north, vehicle.velocity_east,
                  command.north_m_s, command.east_m_s,
                  *(v for p in points for v in (p.north_m, p.east_m)))
        if not all(math.isfinite(v) for v in values):
            return ZoneDecision(False, 'ZONE_INVALID_DATA')
        if not 0 <= now-field.observed_at <= self.max_map_age:
            return ZoneDecision(False, 'ZONE_STALE')
        if not 0 <= now-vehicle.sampled_at <= .3:
            return ZoneDecision(False, 'ZONE_STALE_VEHICLE')
        area = sum(a.north_m*b.east_m-b.north_m*a.east_m for a, b in zip(points, points[1:]+points[:1]))
        if abs(area) < 1e-8:
            return ZoneDecision(False, 'ZONE_INVALID_POLYGON')
        sign = 1 if area > 0 else -1
        clearance = float('inf')
        for a, b in zip(points, points[1:]+points[:1]):
            dn, de = b.north_m-a.north_m, b.east_m-a.east_m
            length = math.hypot(dn, de)
            if length < 1e-8:
                return ZoneDecision(False, 'ZONE_INVALID_POLYGON')
            def side(p):
                return sign*(dn*(p.east_m-a.east_m)-de*(p.north_m-a.north_m))/length
            if any(side(p) < -1e-8 for p in points):
                return ZoneDecision(False, 'ZONE_NONCONVEX')
            clearance = min(clearance, side(vehicle.position))
        # Páska: každá čára je další hrana konvexní oblasti (polorovina se startem).
        for line in field.forbidden_lines:
            if not all(math.isfinite(v) for v in (line.a.north_m, line.a.east_m, line.b.north_m, line.b.east_m)):
                return ZoneDecision(False, 'ZONE_INVALID_DATA')
        tape = field.line_clearance(vehicle.position)
        if tape < 0:
            return ZoneDecision(False, 'ZONE_BEYOND_TAPE', tape)
        clearance = min(clearance, tape)
        # Obal brzdné dráhy ve všech směrech: nezávisí na natočení kamery.
        speed = max(math.hypot(vehicle.velocity_north, vehicle.velocity_east),
                    math.hypot(command.north_m_s, command.east_m_s))
        required = self.margin + speed*self.reaction_time + speed*speed/(2*self.acceleration)
        if clearance < 0:
            return ZoneDecision(False, 'ZONE_OUTSIDE', clearance, required)
        allowed = clearance >= required
        return ZoneDecision(allowed, 'ZONE_CLEAR' if allowed else 'ZONE_BRAKE', clearance, required)
