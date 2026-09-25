"""Kinematický backend celé mise. Neimportuje GPIO ani MAVLink."""
import math
from dataclasses import asdict
from .approach import TargetEstimate, VehicleState
from .field import FieldMap, GroundPoint
from .mission_controller import MissionController, MissionInput, CommandFeedback
from .tape_mapper import TapeMapper

# Zorné pole OV5647 (geolocation.NOMINAL_FOV), kamera kolmo dolů, horní okraj obrazu = sever.
HALF_FOV_DEG = (41.4/2, 53.5/2)


def footprint(height):
    """Poloviční rozměr záběru na zemi (sever, východ) v metrech."""
    return tuple(max(0., height)*math.tan(math.radians(a)) for a in HALF_FOV_DEG)


def visible_tape(tape, n, e, height, samples=60):
    """Část úseku pásky uvnitř záběru kamery, nebo None."""
    half_n, half_e = footprint(height)
    (a_n, a_e), (b_n, b_e) = tape
    inside = [t/samples for t in range(samples+1)
              if abs(a_n+(b_n-a_n)*t/samples-n) <= half_n and abs(a_e+(b_e-a_e)*t/samples-e) <= half_e]
    if len(inside) < 2:
        return None
    t0, t1 = inside[0], inside[-1]
    return (GroundPoint(a_n+(b_n-a_n)*t0, a_e+(b_e-a_e)*t0), GroundPoint(a_n+(b_n-a_n)*t1, a_e+(b_e-a_e)*t1))


class DriftModel:
    """Pomalé ujíždění polohy z GPS/EKF: součet sinusovek s periodami 150-600 s
    (typické bloudění běžného GPS modulu bez RTK: desítky cm za minutu).

    amplitude_m ~ směrodatná odchylka v každé ose. V čase 0 je drift nulový
    (start se zachytí přesně na zelené tečce).
    """
    def __init__(self, amplitude_m=0., seed=1):
        import random
        rng = random.Random(seed)
        self.terms = [[(amplitude_m*math.sqrt(2/3), rng.uniform(150, 600), rng.uniform(0, 2*math.pi))
                       for _ in range(3)] for _ in range(2)]

    def raw(self, t, axis):
        return sum(a*math.sin(2*math.pi*t/p+ph) for a, p, ph in self.terms[axis])

    def __call__(self, t):
        return self.raw(t, 0)-self.raw(0, 0), self.raw(t, 1)-self.raw(0, 1)


def simulate_mission(settings, *, seconds=300., dt=.05, no_target=False, moving=False,
                     fault=None, fault_at=20., target_loss=None, tapes=(), target_at=(8., 5.),
                     camera_period=.1, gps_drift_m=0., camera_noise_m=0., seed=1, green_visible=True):
    """n, e jsou skutečná poloha; řadič dostává polohu zatíženou driftem GPS/EKF.

    Kamera měří relativní posun bodu vůči dronu (se šumem camera_noise_m), takže
    její měření jsou ve stejné (ujeté) soustavě jako poloha dronu - jako ve skutečnosti.
    Zelená tečka je v bodě (0, 0), kde dron startuje.
    """
    if not all(math.isfinite(v) for v in (seconds, dt, fault_at)) or not 0 < seconds <= 3600 or not 0 < dt <= .2:
        raise ValueError('Neplatný čas simulace.')
    if fault not in (None, 'telemetry', 'map', 'camera', 'camera_unlocked', 'manual', 'stop', 'reject_arm', 'no_climb'):
        raise ValueError('Neznámý scénář poruchy.')
    if target_loss is not None and (len(target_loss) != 2 or
            not all(math.isfinite(t) for t in target_loss) or not 0 <= target_loss[0] < target_loss[1]):
        raise ValueError('Neplatný interval ztráty cíle.')
    import random
    rng = random.Random(seed+1000)
    drift = DriftModel(gps_drift_m, seed)
    noise = (lambda: (rng.gauss(0, camera_noise_m), rng.gauss(0, camera_noise_m))) if camera_noise_m else (lambda: (0., 0.))
    mission = MissionController(settings)
    mapper = TapeMapper()
    next_frame = 0.
    n = e = height = vn = ve = up = 0.
    guided = armed = taking_off = False
    feedback = None
    camera_id = None
    camera_since = None
    rows = []
    for i in range(int(seconds/dt)+1):
        now = i*dt
        dn, de = drift(now)
        # n, e = poloha v soustavě EKF, kterou autopilot řídí; skutečná poloha = EKF - drift.
        tn, te = n-dn, e-de
        n0, n1, e0, e1 = settings.bounds
        # Kamera dává snímky po camera_period; páska se promítá jen z viditelné části.
        if now >= next_frame-1e-9:
            next_frame = now+camera_period
            if height >= 1.:
                seen = [v for v in (visible_tape(t, tn, te, height) for t in tapes) if v is not None]
                mapper.update([tuple(GroundPoint(p.north_m+dn, p.east_m+de) for p in v) for v in seen], now)
        field = FieldMap(tuple(GroundPoint(*p) for p in ((n0, e0), (n1, e0), (n1, e1), (n0, e1))), now,
                         mapper.lines)
        active_fault = fault if now >= fault_at else None
        if active_fault == 'map':
            field = FieldMap()
        target_point = GroundPoint(target_at[0], target_at[1]+(2*math.sin(now*.2) if moving else 0))
        half_n, half_e = footprint(height)
        visible = (height >= settings.height_m-.2 and not no_target and
                   abs(target_point.north_m-tn) <= half_n and abs(target_point.east_m-te) <= half_e and
                   not (target_loss and target_loss[0] <= now < target_loss[1]))
        def measured(point):
            en, ee = noise()
            return GroundPoint(point.north_m+dn+en, point.east_m+de+ee)
        target = TargetEstimate(measured(target_point), now, 'simulated-dot', .05) if visible else None
        green_seen = (green_visible and height >= 1. and abs(tn) <= half_n and abs(te) <= half_e)
        green = TargetEstimate(measured(GroundPoint(0., 0.)), now, 'green', .05) if green_seen else None
        camera_locked = bool(visible and camera_id == 'simulated-dot' and
                             camera_since is not None and now-camera_since >= .5 and active_fault != 'camera_unlocked')
        data = MissionInput(VehicleState(GroundPoint(n, e), vn, ve,
                            now-1 if active_fault == 'telemetry' else now),
                            field, height, up, now, guided, armed, height < .1,
                            active_fault != 'camera', camera_locked, target, feedback,
                            active_fault == 'manual', active_fault == 'stop', green)
        command = mission.update(data, now)
        rows.append(dict(time_s=now, backend='kinematic_simulation',
                         north_m=tn, east_m=te, height_m=height, drift_north_m=dn, drift_east_m=de,
                         green_visible=green is not None,
                         velocity_north_m_s=vn, velocity_east_m_s=ve, velocity_up_m_s=up,
                         target_visible=visible, camera_locked=camera_locked,
                         tape_lines=[[l.a.north_m, l.a.east_m, l.b.north_m, l.b.east_m] for l in mapper.lines],
                         distance_m=math.hypot(target_point.north_m-tn, target_point.east_m-te),
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
        if command.action == 'LAND':
            climb, taking_off = -.5, False
            desired_n = desired_e = 0.
        elif command.action in ('BRAKE', 'RELEASE') or fault == 'no_climb':
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
