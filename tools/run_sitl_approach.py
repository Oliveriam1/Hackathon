"""Integrační test lokálního ArduPilot SITL; vzlet a GUIDED nastavte v simulátoru.

Nevstupuje do běžné kamerové pipeline. Cíl je syntetický bod v metrech od
počáteční polohy simulátoru. Vyžaduje zprávu SIMSTATE ze stejného autopilota.
"""
import argparse
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.approach import ApproachController, TargetEstimate, VehicleState
from src.field import GroundPoint
from src.sitl_velocity import SitlVelocityAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5760)
    parser.add_argument('--target', nargs=2, type=float, default=(8., 5.), metavar=('NORTH', 'EAST'))
    parser.add_argument('--seconds', type=float, default=30.)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not math.isfinite(args.seconds) or not 0 < args.seconds <= 300:
        parser.error('Neplatný port nebo délka testu.')
    if not all(math.isfinite(v) and abs(v) < 19 for v in args.target):
        parser.error('Cíl musí být uvnitř simulační zóny ±19 m.')
    try:
        from pymavlink import mavutil
    except ImportError:
        parser.error('SITL test vyžaduje pymavlink; lokální simulace tuto závislost nepotřebuje.')
    connection = mavutil.mavlink_connection(f'tcp:127.0.0.1:{args.port}', source_system=245)
    adapter = None
    guided = False
    try:
        heartbeat = connection.wait_heartbeat(timeout=10)
        if heartbeat is None or heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            raise RuntimeError('Chybí heartbeat ArduPilotu.')
        system, component = heartbeat.get_srcSystem(), heartbeat.get_srcComponent()
        connection.target_system, connection.target_component = system, component
        for message_id in (32, 164):  # LOCAL_POSITION_NED, SIMSTATE
            connection.mav.command_long_send(system, component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, message_id, 100000, 0, 0, 0, 0, 0)
        until = time.monotonic()+10
        seen_sim = False
        position = None
        last_position = last_heartbeat = 0.
        while time.monotonic() < until:
            msg = connection.recv_match(blocking=True, timeout=.1)
            if msg is None or (msg.get_srcSystem(), msg.get_srcComponent()) != (system, component):
                continue
            if msg.get_type() == 'SIMSTATE': seen_sim = True
            if msg.get_type() == 'LOCAL_POSITION_NED':
                position, last_position = msg, time.monotonic()
            if msg.get_type() == 'HEARTBEAT':
                heartbeat, last_heartbeat = msg, time.monotonic()
            if seen_sim and position is not None and last_heartbeat:
                break
        if not seen_sim or position is None:
            raise RuntimeError('Nepotvrzený SITL: chybí SIMSTATE nebo lokální poloha.')
        guided_mode = connection.mode_mapping().get('GUIDED')
        def ready(hb):
            return (guided_mode is not None and hb.custom_mode == guided_mode and
                    bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
        if not ready(heartbeat) or not position.z < -1.:
            raise RuntimeError('V SITL nejprve proveďte vzlet a nastavte GUIDED; test sám nearmuje.')
        origin = (position.x, position.y)
        adapter = SitlVelocityAdapter(connection, simulator_confirmed=True)
        controller = ApproachController()
        start = last_print = time.monotonic()
        while time.monotonic()-start < args.seconds:
            # Omezený počet zpráv: zahlcení streamem nesmí blokovat kontrolní smyčku.
            for _ in range(200):
                msg = connection.recv_match(blocking=False)
                if msg is None: break
                if (msg.get_srcSystem(), msg.get_srcComponent()) != (system, component): continue
                if msg.get_type() == 'LOCAL_POSITION_NED':
                    position, last_position = msg, time.monotonic()
                if msg.get_type() == 'HEARTBEAT':
                    heartbeat, last_heartbeat = msg, time.monotonic()
            now = time.monotonic()
            guided = ready(heartbeat)
            if not guided:
                raise RuntimeError('Simulátor opustil armed GUIDED; test končí.')
            if now-last_heartbeat > 2:
                adapter.hold()
                raise RuntimeError('Výpadek heartbeat simulátoru.')
            state = VehicleState(GroundPoint(position.x-origin[0], position.y-origin[1]),
                                 position.vx, position.vy, last_position)
            target = TargetEstimate(GroundPoint(*args.target), now, 'synthetic-sitl-dot')
            command = controller.update(state, target, now)
            adapter.send(command)
            if now-last_print >= 1:
                print(command, flush=True)
                last_print = now
            time.sleep(.05)
    finally:
        try:
            if adapter is not None and guided:
                adapter.hold()
        finally:
            connection.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as error:
        print(f'SITL test: {error}', file=sys.stderr)
        raise SystemExit(1)
