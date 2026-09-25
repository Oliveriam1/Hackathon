"""Společný řadič mise: čisté záměry a stav, žádný přístup k hardwaru."""
import math
from dataclasses import dataclass
from .approach import ApproachCommand, TargetEstimate, VehicleState
from .config import StartReference
from .field import FieldMap
from .search_mission import SearchMission
from .search_planner import RectangleSweep
from .takeoff import TakeoffController
from .zone_guard import ZoneGuard


@dataclass(frozen=True)
class MissionSettings:
    start: StartReference
    bounds: tuple[float, float, float, float]
    height_m: float = 5.
    lane_spacing_m: float = 3.
    timeout_s: float = 600.
    # Při kameře kolmo dolů vidí dron pásku jen ~1.9 m dopředu (5 m, OV5647).
    # Rychlost musí dovolit zastavit před ní: rezerva + reakce + brzdná dráha.
    max_speed_m_s: float = 1.

    def __post_init__(self):
        if (not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or
                not math.isfinite(self.start.latitude_deg) or abs(self.start.latitude_deg) >= 89.9 or
                not math.isfinite(self.start.longitude_deg)):
            raise ValueError('Neplatná reference nebo timeout mise.')
        if not math.isfinite(self.max_speed_m_s) or not 0 < self.max_speed_m_s <= 2:
            raise ValueError('Rychlost mise musí být v rozsahu (0, 2] m/s.')
        TakeoffController(self.height_m)
        RectangleSweep(self.bounds, spacing=self.lane_spacing_m)


@dataclass(frozen=True)
class CommandFeedback:
    request_id: int
    accepted: bool


@dataclass(frozen=True)
class MissionInput:
    vehicle: VehicleState
    field: FieldMap
    height_m: float
    up_m_s: float
    heartbeat_at: float
    guided: bool = False
    armed: bool = False
    on_ground: bool = True
    camera_ready: bool = False
    camera_locked: bool = False
    target: TargetEstimate | None = None
    feedback: CommandFeedback | None = None
    manual_override: bool = False
    stop_requested: bool = False


@dataclass(frozen=True)
class MissionCommand:
    state: str
    reason: str
    action: str = 'HOLD'
    request_id: int | None = None
    north_m_s: float = 0.
    east_m_s: float = 0.
    up_m_s: float = 0.
    height_setpoint_m: float | None = None
    camera_action: str = 'HOLD'
    target_id: str | None = None


# Kratší polovina záběru OV5647 kolmo dolů (vertikální FOV 41.4°).
TAPE_LOOKAHEAD_TAN = math.tan(math.radians(41.4/2))


def tape_safe_speed(height_m, *, margin=.5, reaction_s=.5, acceleration=1.):
    """Nejvyšší rychlost, při které dron zastaví před páskou, kterou uvidí až na okraji záběru.

    reaction_s zahrnuje reakci autopilota i potvrzení čáry ze dvou snímků.
    """
    lookahead = height_m*TAPE_LOOKAHEAD_TAN
    room = lookahead-margin
    if room <= 0:
        return 0.
    ar = acceleration*reaction_s
    return math.sqrt(ar*ar+2*acceleration*room)-ar


class MissionController:
    def __init__(self, settings=None):
        self.settings = settings
        self.state = 'PREFLIGHT' if settings else 'DISABLED'
        self.reason = 'WAITING_FOR_INPUT' if settings else 'NO_FLIGHT_BACKEND'
        self.started = self.last_time = self.phase_started = None
        self.pending = None
        self.acknowledged = False
        self.request_sequence = 0
        self.last_result = None
        self.zone_state = 'ZONE_UNKNOWN'
        self.search = SearchMission(RectangleSweep(settings.bounds, spacing=settings.lane_spacing_m)) if settings else None
        self.takeoff = TakeoffController(settings.height_m) if settings else None
        self.guard = ZoneGuard()
        self.airborne = False
        self.last_tape_lines = ()
        self.speed_cap = (min(settings.max_speed_m_s, max(.2, tape_safe_speed(settings.height_m)))
                          if settings else None)

    def snapshot(self):
        return dict(enabled=self.settings is not None, state=self.state, reason=self.reason,
                    takeoff_complete=self.airborne, zone_state=self.zone_state,
                    waypoint_index=self.search.index if self.search else None,
                    waypoint_count=len(self.search.sweep.points) if self.search else None,
                    target_result=self.last_result, flight_ready=False,
                    tape_lines=len(self.last_tape_lines), speed_cap_m_s=self.speed_cap,
                    execution='controller_only_no_hardware')

    def output(self, reason, **kwargs):
        self.reason = reason
        return MissionCommand(self.state, reason, **kwargs)

    def fail(self, reason):
        self.state = 'FAILSAFE'
        self.pending = None
        self.search.controller.hold(reason)
        return self.output(reason, action='BRAKE')

    def request(self, action, now):
        self.request_sequence += 1
        self.pending = (self.request_sequence, action)
        self.acknowledged = False
        self.phase_started = now
        return self.output('WAITING_FOR_ACK_AND_STATE', action=action,
                           request_id=self.request_sequence, height_setpoint_m=self.settings.height_m)

    def update(self, data, now):
        if self.settings is None:
            return self.output('NO_FLIGHT_BACKEND', action='NONE')
        if data.manual_override or self.state == 'MANUAL':
            self.state = 'MANUAL'
            return self.output('OPERATOR_CONTROL', action='RELEASE')
        if data.stop_requested or self.state == 'ABORTED':
            self.state = 'ABORTED'
            return self.output('STOP_REQUESTED', action='BRAKE')
        if self.state == 'FAILSAFE':
            return self.output(self.reason, action='BRAKE')
        if (not math.isfinite(now) or
                self.last_time is not None and not 0 < now-self.last_time <= .3):
            return self.fail('CONTROL_CLOCK_INVALID_OR_GAP')
        self.last_time = now
        values = (data.height_m, data.up_m_s, data.heartbeat_at, data.vehicle.sampled_at,
                  data.vehicle.position.north_m, data.vehicle.position.east_m,
                  data.vehicle.velocity_north, data.vehicle.velocity_east)
        healthy = (all(math.isfinite(v) for v in values) and
                   0 <= now-data.heartbeat_at <= 2 and 0 <= now-data.vehicle.sampled_at <= .3)
        if not healthy:
            return self.output('WAITING_FOR_TELEMETRY') if self.state == 'PREFLIGHT' else self.fail('TELEMETRY_INVALID_OR_STALE')
        if self.started is not None:
            if now-self.started > self.settings.timeout_s:
                return self.fail('MISSION_TIMEOUT')
            if self.state != 'SETTING_GUIDED' and not data.guided:
                return self.fail('GUIDED_LOST')
            if self.state not in ('SETTING_GUIDED', 'ARMING') and not data.armed:
                return self.fail('ARMED_LOST')
            if not data.camera_ready:
                return self.fail('CAMERA_UNAVAILABLE')
        self.last_tape_lines = data.field.forbidden_lines
        zone = self.guard.check(data.field, data.vehicle, ApproachCommand('HOLD'), now)
        self.zone_state = zone.state
        if not zone.allowed:
            if self.state == 'PREFLIGHT':
                return self.output(zone.state)
            # Brzdění je obnovitelné; neznámá/stará mapa nebo opuštění pole je chyba.
            if zone.state == 'ZONE_BEYOND_TAPE' and self.airborne:
                # Páska rozpoznaná až za dronem: pomalu zpět kolmo k ní, ne failsafe.
                line = min(data.field.forbidden_lines, key=lambda l: l.signed_distance(data.vehicle.position))
                dn, de = line.inward()
                self.search.controller.hold(zone.state)
                self.search.controller.last_time = now
                self.state = 'TAPE_RETREAT'
                return self.output(zone.state, action='VELOCITY', north_m_s=.5*dn, east_m_s=.5*de,
                                   height_setpoint_m=self.settings.height_m)
            if zone.state == 'ZONE_BRAKE' and self.airborne:
                self.search.controller.hold(zone.state)
                self.search.controller.last_time = now
                self.state = 'ZONE_HOLD'
                return self.output(zone.state, action='BRAKE')
            return self.fail(zone.state)
        n0, n1, e0, e1 = self.settings.bounds
        expected = {(n0, e0), (n1, e0), (n1, e1), (n0, e1)}
        if len(data.field.verified_boundary) != 4 or {(p.north_m, p.east_m) for p in data.field.verified_boundary} != expected:
            return self.output('MAP_ROUTE_MISMATCH') if self.state == 'PREFLIGHT' else self.fail('MAP_ROUTE_MISMATCH')
        if not data.camera_ready:
            return self.output('WAITING_FOR_CAMERA') if self.state == 'PREFLIGHT' else self.fail('CAMERA_UNAVAILABLE')
        if self.state == 'PREFLIGHT':
            if data.armed or not data.on_ground or abs(data.height_m) > .3 or abs(data.up_m_s) > .15:
                return self.output('START_REQUIRES_DISARMED_ON_GROUND')
            if math.hypot(data.vehicle.position.north_m, data.vehicle.position.east_m) > .5:
                return self.output('NOT_AT_START_REFERENCE')
            self.started = now
            self.state = 'SETTING_GUIDED'
            return self.request('SET_GUIDED', now)
        if now-self.started > self.settings.timeout_s:
            return self.fail('MISSION_TIMEOUT')
        if self.pending:
            if data.feedback and data.feedback.request_id == self.pending[0]:
                if not data.feedback.accepted:
                    return self.fail('COMMAND_REJECTED')
                self.acknowledged = True
            if not self.acknowledged and now-self.phase_started > 10:
                return self.fail('COMMAND_ACK_TIMEOUT')
        if self.state == 'SETTING_GUIDED':
            if self.acknowledged and data.guided:
                self.state = 'ARMING'
                return self.request('ARM', now)
            if now-self.phase_started > 15:
                return self.fail('MODE_TIMEOUT')
            return self.output('WAITING_FOR_GUIDED', action='WAIT')
        if not data.guided:
            return self.fail('GUIDED_LOST')
        if self.state == 'ARMING':
            if self.acknowledged and data.armed:
                self.state = 'TAKING_OFF'
                return self.request('TAKEOFF', now)
            if now-self.phase_started > 15:
                return self.fail('ARM_TIMEOUT')
            return self.output('WAITING_FOR_ARMED', action='WAIT')
        if not data.armed:
            return self.fail('ARMED_LOST')
        if not self.airborne:
            if not self.acknowledged:
                return self.output('WAITING_FOR_TAKEOFF_ACK', action='WAIT')
            result = self.takeoff.update(data.height_m, data.up_m_s, data.vehicle.sampled_at, now)
            if self.takeoff.failure:
                return self.fail(result.state)
            if not result.ready:
                self.state = result.state
                return self.output('WAITING_FOR_HEIGHT', action='TAKEOFF_MONITOR',
                                   up_m_s=result.up_m_s, height_setpoint_m=self.settings.height_m,
                                   camera_action='SEARCH')
            self.pending = None
            self.airborne = True
            self.state = 'SEARCHING'
        if data.on_ground or abs(data.height_m-self.settings.height_m) > .75:
            return self.fail('FLIGHT_HEIGHT_LOST')
        # Zpomalovat předem podle volného prostoru, ne opakovaně rozjíždět/brzdit.
        room = max(0., zone.clearance_m-self.guard.margin-.15)
        ar = self.guard.acceleration*self.guard.reaction_time
        speed_limit = math.sqrt(ar*ar+2*self.guard.acceleration*room)-ar
        self.search.controller.speed = min(self.speed_cap, max(.01, speed_limit))
        command = self.search.update(data.vehicle, data.target, now, data.field)
        zone = self.guard.check(data.field, data.vehicle, command, now)
        self.zone_state = zone.state
        if not zone.allowed:
            self.search.controller.hold(zone.state)
            self.search.controller.last_time = now
            self.state = 'ZONE_HOLD'
            return self.output(zone.state, action='BRAKE')
        states = {'ACQUIRING_TARGET': 'ACQUIRING_TARGET', 'APPROACHING': 'APPROACHING',
                  'SETTLING': 'CENTERING', 'ARRIVED': 'HOLDING_TARGET',
                  'SEARCHING': 'SEARCHING', 'SEARCH_COMPLETE': 'SEARCH_COMPLETE',
                  'TARGET_LOST_HOLD': 'TARGET_LOST_HOLD'}
        if command.state not in states:
            return self.fail(command.state)
        self.state = states[command.state]
        tracking = self.search.tracking and data.target is not None and command.state != 'TARGET_LOST_HOLD'
        if self.state == 'HOLDING_TARGET' and not data.camera_locked:
            self.state = 'CENTERING_CAMERA'
        if self.state == 'HOLDING_TARGET':
            from .geolocation import offset_to_latlon
            point = data.target.position
            lat, lon = offset_to_latlon(self.settings.start.latitude_deg, self.settings.start.longitude_deg,
                                        point.north_m, point.east_m)
            self.last_result = dict(target_id=data.target.identity, latitude_deg=lat,
                                    longitude_deg=(lon+180)%360-180, north_m=point.north_m,
                                    east_m=point.east_m, sampled_at=data.target.sampled_at,
                                    horizontal_error_estimate_m=data.target.error_m,
                                    datum='WGS84', historical_observation=True)
        return self.output(command.state, action='VELOCITY', north_m_s=command.north_m_s,
                           east_m_s=command.east_m_s, height_setpoint_m=self.settings.height_m,
                           camera_action='TRACK' if tracking else 'SEARCH',
                           target_id=data.target.identity if tracking else None)
