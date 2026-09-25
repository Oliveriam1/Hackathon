"""Lokální kinematická simulace; žádné importy MAVLink ani přístup k zařízení."""
import math
from .approach import ApproachController, ApproachCommand, TargetEstimate, VehicleState
from .field import GroundPoint
from .takeoff import TakeoffController


def simulate(*, seconds=25., dt=.05, moving=False, loss_start=None, loss_end=None, takeoff_height=None):
    if not math.isfinite(dt) or not 0 < dt <= .2 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Neplatná délka simulace nebo časový krok.')
    controller = ApproachController()
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
        estimate = None if lost else TargetEstimate(target, now, 'simulated-dot')
        climb = 0.
        if not airborne:
            result = takeoff.update(height, up, now, now)
            climb = result.up_m_s
            airborne = result.ready
            cmd = ApproachCommand(result.state)
        else:
            cmd = controller.update(VehicleState(position, vn, ve, now), estimate, now)
            if takeoff:
                climb = max(-.5, min(1., (takeoff.height-height)*.8))
        rows.append(dict(time_s=now, north_m=position.north_m, east_m=position.east_m,
                         height_m=height if takeoff else None, velocity_up_m_s=up,
                         command_up_m_s=climb, takeoff_complete=airborne,
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
