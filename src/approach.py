"""Přelet k bodu v lokální rovině, zatím používaný pouze simulátorem."""
from dataclasses import dataclass
import math
from .field import GroundPoint


@dataclass(frozen=True)
class TargetEstimate:
    position: GroundPoint
    sampled_at: float
    identity: str
    error_m: float = 0.
    valid: bool = True


@dataclass(frozen=True)
class VehicleState:
    position: GroundPoint
    velocity_north: float
    velocity_east: float
    sampled_at: float


@dataclass(frozen=True)
class ApproachCommand:
    state: str
    north_m_s: float = 0.
    east_m_s: float = 0.
    distance_m: float | None = None


class ApproachController:
    def __init__(self, *, bounds=(-20., 20., -20., 20.), speed=2., acceleration=1.,
                 tolerance=.4, dwell=1., max_error=2.):
        if not (all(math.isfinite(v) for v in (*bounds, speed, acceleration, tolerance, dwell, max_error))
                and max_error >= 0 and 0 < speed and 0 < acceleration and 0 < tolerance and 0 < dwell and
                len(bounds) == 4 and bounds[0] < bounds[1] and bounds[2] < bounds[3]):
            raise ValueError('Neplatné parametry přeletu.')
        self.bounds, self.speed, self.acceleration = bounds, speed, acceleration
        self.tolerance, self.dwell, self.max_error = tolerance, dwell, max_error
        self.last_time = self.centered_since = self.identity = None
        self.previous = (0., 0.)

    def hold(self, state):
        self.centered_since = None
        self.previous = (0., 0.)
        return ApproachCommand(state)

    def inside(self, point, margin=0.):
        n0, n1, e0, e1 = self.bounds
        return n0+margin <= point.north_m <= n1-margin and e0+margin <= point.east_m <= e1-margin

    def update(self, vehicle, target, now):
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            return self.hold('INVALID_TIME')
        gap = self.last_time is not None and now-self.last_time > .3
        dt = .05 if self.last_time is None else min(.2, now-self.last_time)
        self.last_time = now
        if gap:
            return self.hold('STREAM_GAP')
        if not all(math.isfinite(v) for v in (vehicle.position.north_m, vehicle.position.east_m,
                   vehicle.velocity_north, vehicle.velocity_east, vehicle.sampled_at)):
            return self.hold('INVALID_TELEMETRY')
        if not 0 <= now-vehicle.sampled_at <= .3:
            return self.hold('STALE_TELEMETRY')
        if target is None or not target.valid:
            return self.hold('TARGET_LOST')
        if not all(math.isfinite(v) for v in (target.position.north_m, target.position.east_m,
                                              target.sampled_at, target.error_m)):
            return self.hold('INVALID_TARGET')
        if not 0 <= now-target.sampled_at <= .25:
            return self.hold('STALE_TARGET')
        if not 0 <= target.error_m <= self.max_error:
            return self.hold('UNCERTAIN_TARGET')
        if not self.inside(vehicle.position) or not self.inside(target.position, target.error_m+self.tolerance):
            return self.hold('ZONE_LIMIT')
        # Brzdná dráha vozidla musí zůstat uvnitř syntetické obdélníkové zóny.
        vn, ve = vehicle.velocity_north, vehicle.velocity_east
        speed = math.hypot(vn, ve)
        stopping = GroundPoint(vehicle.position.north_m+vn*speed/(2*self.acceleration),
                               vehicle.position.east_m+ve*speed/(2*self.acceleration))
        if not self.inside(stopping, .1):
            return self.hold('ZONE_BRAKING')
        if self.identity != target.identity:
            self.identity = target.identity
            return self.hold('ACQUIRING_TARGET')
        dn = target.position.north_m-vehicle.position.north_m
        de = target.position.east_m-vehicle.position.east_m
        distance = math.hypot(dn, de)
        if distance <= self.tolerance and speed < .15:
            if self.centered_since is None:
                self.centered_since = now
            self.previous = (0., 0.)
            return ApproachCommand('ARRIVED' if now-self.centered_since >= self.dwell else 'SETTLING',
                                   distance_m=distance)
        self.centered_since = None
        desired_speed = min(self.speed, .8*distance,
                            math.sqrt(2*self.acceleration*max(0., distance-self.tolerance/2)))
        desired = (dn/distance*desired_speed, de/distance*desired_speed) if distance else (0., 0.)
        dx, dy = desired[0]-self.previous[0], desired[1]-self.previous[1]
        change = math.hypot(dx, dy)
        factor = min(1., self.acceleration*dt/change) if change else 1.
        self.previous = (self.previous[0]+dx*factor, self.previous[1]+dy*factor)
        return ApproachCommand('APPROACHING', *self.previous, distance)
