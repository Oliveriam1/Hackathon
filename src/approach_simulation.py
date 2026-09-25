"""Lokální kinematická simulace; žádné importy MAVLink ani přístup k zařízení."""
import math
from .approach import ApproachController, ApproachCommand, TargetEstimate, VehicleState
from .field import GroundPoint, FieldMap
from .zone_guard import ZoneGuard
from .takeoff import TakeoffController
from .search_planner import RectangleSweep
from .search_mission import SearchMission


def simulate(*, seconds=25., dt=.05, moving=False, loss_start=None, loss_end=None, takeoff_height=None,
             search=False, field_bounds=(-10., 10., -10., 10.), lane_spacing=4., sensor_radius=3., no_target=False):
    if not math.isfinite(dt) or not 0 < dt <= .2 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Neplatná délka simulace nebo časový krok.')
    controller = ApproachController()
    if not math.isfinite(sensor_radius) or sensor_radius <= 0:
        raise ValueError('Dosah simulační detekce musí být kladný.')
    mission = SearchMission(RectangleSweep(field_bounds, spacing=lane_spacing)) if search else None
    guard = ZoneGuard()
    if mission and not mission.controller.inside(GroundPoint(0., 0.), .5):
        raise ValueError('Start (0, 0) musí ležet uvnitř pole s rezervou.')
    takeoff = TakeoffController(takeoff_height) if takeoff_height is not None else None
    height = up = 0.
    airborne = takeoff is None
    position = GroundPoint(0., 0.)
    vn = ve = 0.
    rows = []
    for i in range(int(seconds/dt)+1):
        now = i*dt
        target = GroundPoint(8., 5.+(math.sin(now*.2)*2 if moving else 0.))
        lost = loss_start is not None and loss_start <= now < loss_end
        visible = not (lost or no_target) and (not search or math.hypot(
            target.north_m-position.north_m, target.east_m-position.east_m) <= sensor_radius)
        estimate = TargetEstimate(target, now, 'simulated-dot') if visible else None
        climb = 0.
        if not airborne:
            result = takeoff.update(height, up, now, now)
            climb = result.up_m_s
            airborne = result.ready
            cmd = ApproachCommand(result.state)
        else:
            cmd = (mission or controller).update(VehicleState(position, vn, ve, now), estimate, now)
            if takeoff:
                climb = max(-.5, min(1., (takeoff.height-height)*.8))
        zone = None
        if mission:
            n0, n1, e0, e1 = field_bounds
            # Mapa pochází ze syntetického světa, nikoliv z rozpoznání pásky.
            field = FieldMap((GroundPoint(n0, e0), GroundPoint(n1, e0),
                              GroundPoint(n1, e1), GroundPoint(n0, e1)), now)
            zone = guard.check(field, VehicleState(position, vn, ve, now), cmd, now)
            if not zone.allowed:
                cmd = mission.controller.hold(zone.state)
        rows.append(dict(time_s=now, north_m=position.north_m, east_m=position.east_m,
                         zone_state=zone.state if zone else 'NOT_CHECKED',
                         zone_source='synthetic_field' if zone else None,
                         boundary_clearance_m=zone.clearance_m if zone else None,
                         height_m=height if takeoff else None, velocity_up_m_s=up,
                         command_up_m_s=climb, takeoff_complete=airborne,
                         target_visible=visible, search_waypoint=mission.index if mission else None,
                         search_waypoint_count=len(mission.sweep.points) if mission else None,
                         target_north_m=target.north_m, target_east_m=target.east_m,
                         velocity_north_m_s=vn, velocity_east_m_s=ve,
                         command_north_m_s=cmd.north_m_s, command_east_m_s=cmd.east_m_s,
                         distance_m=math.hypot(target.north_m-position.north_m, target.east_m-position.east_m),
                         state=cmd.state, backend='kinematic_simulation'))
        # Konečná schopnost brzdit: nulový povel neznamená okamžité zastavení.
        dvn, dve = cmd.north_m_s-vn, cmd.east_m_s-ve
        delta = math.hypot(dvn, dve)
        factor = min(1., controller.acceleration*dt/delta) if delta else 1.
        vn, ve = vn+dvn*factor, ve+dve*factor
        position = GroundPoint(position.north_m+vn*dt, position.east_m+ve*dt)
        up += max(-dt, min(dt, climb-up))
        height = max(0., height+up*dt)
    return rows
