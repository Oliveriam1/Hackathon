"""Kinematický backend celé mise. Neimportuje GPIO ani MAVLink."""
import math
from dataclasses import asdict
from .approach import TargetEstimate, VehicleState
from .field import FieldMap, GroundPoint
from .mission_controller import MissionController, MissionInput, CommandFeedback


def simulate_mission(settings, *, seconds=300., dt=.05, no_target=False, moving=False,
                     fault=None, fault_at=20., target_loss=None):
    if not all(math.isfinite(v) for v in (seconds, dt, fault_at)) or not 0 < seconds <= 3600 or not 0 < dt <= .2:
        raise ValueError('Neplatný čas simulace.')
    if fault not in (None, 'telemetry', 'map', 'camera', 'camera_unlocked', 'manual', 'stop', 'reject_arm', 'no_climb'):
        raise ValueError('Neznámý scénář poruchy.')
    if target_loss is not None and (len(target_loss) != 2 or
            not all(math.isfinite(t) for t in target_loss) or not 0 <= target_loss[0] < target_loss[1]):
        raise ValueError('Neplatný interval ztráty cíle.')
    mission = MissionController(settings)
    n = e = height = vn = ve = up = 0.
    guided = armed = taking_off = False
    feedback = None
    camera_id = None
    camera_since = None
    rows = []
    for i in range(int(seconds/dt)+1):
        now = i*dt
        n0, n1, e0, e1 = settings.bounds
        field = FieldMap(tuple(GroundPoint(*p) for p in ((n0, e0), (n1, e0), (n1, e1), (n0, e1))), now)
        active_fault = fault if now >= fault_at else None
        if active_fault == 'map':
            field = FieldMap()
        target_point = GroundPoint(8., 5.+(2*math.sin(now*.2) if moving else 0))
        visible = (height >= settings.height_m-.2 and not no_target and
                   math.hypot(target_point.north_m-n, target_point.east_m-e) <= 3 and
                   not (target_loss and target_loss[0] <= now < target_loss[1]))
        target = TargetEstimate(target_point, now, 'simulated-dot', .05) if visible else None
        camera_locked = bool(visible and camera_id == 'simulated-dot' and
                             camera_since is not None and now-camera_since >= .5 and active_fault != 'camera_unlocked')
        data = MissionInput(VehicleState(GroundPoint(n, e), vn, ve,
                            now-1 if active_fault == 'telemetry' else now),
                            field, height, up, now, guided, armed, height < .1,
                            active_fault != 'camera', camera_locked, target, feedback,
                            active_fault == 'manual', active_fault == 'stop')
        command = mission.update(data, now)
        rows.append(dict(time_s=now, backend='kinematic_simulation',
                         north_m=n, east_m=e, height_m=height,
                         velocity_north_m_s=vn, velocity_east_m_s=ve, velocity_up_m_s=up,
                         target_visible=visible, camera_locked=camera_locked,
                         distance_m=math.hypot(target_point.north_m-n, target_point.east_m-e),
                         command=asdict(command), mission=mission.snapshot()))
        feedback = None
        if command.action in ('SET_GUIDED', 'ARM', 'TAKEOFF'):
            accepted = not (fault == 'reject_arm' and command.action == 'ARM')
            feedback = CommandFeedback(command.request_id, accepted)
            if accepted:
                if command.action == 'SET_GUIDED': guided = True
                if command.action == 'ARM': armed = True
                if command.action == 'TAKEOFF': taking_off = True
        if command.camera_action == 'TRACK':
            if camera_id != command.target_id:
                camera_id, camera_since = command.target_id, now
        else:
            camera_id = camera_since = None
        desired_n, desired_e = command.north_m_s, command.east_m_s
        # Model autopilota udržuje výšku. BRAKE zastaví stoupání, RELEASE značí
        # ruční převzetí; v testu modelujeme následné zastavení pilotem.
        climb = max(-.5, min(1., .8*(settings.height_m-height))) if taking_off and armed else 0.
        if command.action in ('BRAKE', 'RELEASE') or fault == 'no_climb':
            climb = 0.
            if command.action in ('BRAKE', 'RELEASE'):
                taking_off = False
        elif mission.airborne:
            taking_off = True
        delta_n, delta_e = desired_n-vn, desired_e-ve
        delta = math.hypot(delta_n, delta_e)
        factor = min(1., dt/delta) if delta else 1.
        vn, ve = vn+delta_n*factor, ve+delta_e*factor
        n, e = n+vn*dt, e+ve*dt
        up += max(-dt, min(dt, climb-up))
        height = max(0., height+up*dt)
    return rows
