"""Lokální kinematická simulace; žádné importy MAVLink ani přístup k zařízení."""
import math
from .approach import ApproachController, TargetEstimate, VehicleState
from .field import GroundPoint


def simulate(*, seconds=25., dt=.05, moving=False, loss_start=None, loss_end=None):
    controller = ApproachController()
    position = GroundPoint(0., 0.)
    vn = ve = 0.
    rows = []
    for i in range(int(seconds/dt)+1):
        now = i*dt
        target = GroundPoint(8., 5.+(math.sin(now*.2)*2 if moving else 0.))
        lost = loss_start is not None and loss_start <= now < loss_end
        estimate = None if lost else TargetEstimate(target, now, 'simulated-dot')
        cmd = controller.update(VehicleState(position, vn, ve, now), estimate, now)
        rows.append(dict(time_s=now, north_m=position.north_m, east_m=position.east_m,
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
    return rows
