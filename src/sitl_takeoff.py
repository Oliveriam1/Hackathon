"""Jednorázový automatický vzlet výhradně pro potvrzený lokální SITL."""
import math
import time
from .takeoff import TakeoffController


def automatic_takeoff(connection, mavlink, height, *, simulator_confirmed=False,
                      clock=time.monotonic, report=print):
    if not simulator_confirmed:
        raise ValueError('Vzlet vyžaduje potvrzený SITL.')
    monitor = TakeoffController(height)
    system, component = connection.target_system, connection.target_component
    mode = connection.mode_mapping().get('GUIDED')
    if mode is None:
        raise RuntimeError('Simulátor nenabízí GUIDED.')
    heartbeat = position = None
    hb_time = pos_time = -float('inf')

    def receive():
        nonlocal heartbeat, position, hb_time, pos_time
        msg = connection.recv_match(blocking=True, timeout=.1)
        if msg is None or (msg.get_srcSystem(), msg.get_srcComponent()) != (system, component):
            return None
        if msg.get_type() == 'HEARTBEAT':
            heartbeat, hb_time = msg, clock()
        elif msg.get_type() == 'GLOBAL_POSITION_INT':
            position, pos_time = msg, clock()
        return msg

    def fresh():
        return clock()-hb_time <= 2 and clock()-pos_time <= .3

    def armed():
        return bool(heartbeat.base_mode & mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    def command(command_id, params, condition, label):
        report(label)
        connection.mav.command_long_send(system, component, command_id, 0, *params)
        deadline = clock()+15
        accepted = False
        while clock() < deadline:
            msg = receive()
            if msg is not None and msg.get_type() == 'COMMAND_ACK' and msg.command == command_id:
                if msg.result == mavlink.MAV_RESULT_ACCEPTED:
                    accepted = True
                elif msg.result != mavlink.MAV_RESULT_IN_PROGRESS:
                    raise RuntimeError(f'{label}: autopilot odmítl povel ({msg.result}).')
            if not fresh():
                raise RuntimeError(f'{label}: výpadek telemetrie.')
            if accepted and condition():
                return
        raise RuntimeError(f'{label}: timeout potvrzení.')

    # Požadujeme čerstvou výšku vůči home, nikoli výšku vůči počátku EKF.
    deadline = clock()+10
    while clock() < deadline:
        receive()
        if fresh():
            break
    if not fresh():
        raise RuntimeError('Vzlet: chybí čerstvý heartbeat nebo GLOBAL_POSITION_INT.')
    if (not all(math.isfinite(v) for v in (position.relative_alt, position.vz)) or
            armed() or abs(position.relative_alt/1000.) > .3 or abs(position.vz/100.) > .15):
        raise RuntimeError('Automatický SITL vzlet vyžaduje nearmovaný simulátor na zemi u home.')
    command(mavlink.MAV_CMD_DO_SET_MODE,
            (mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode, 0, 0, 0, 0, 0),
            lambda: heartbeat.custom_mode == mode, 'SETTING_GUIDED')
    command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, (1, 0, 0, 0, 0, 0, 0),
            lambda: armed() and heartbeat.custom_mode == mode, 'ARMING')
    command(mavlink.MAV_CMD_NAV_TAKEOFF, (0, 0, 0, 0, 0, 0, height),
            lambda: armed() and heartbeat.custom_mode == mode, 'TAKING_OFF')
    last_sample = None
    while True:
        receive()
        if not fresh() or not armed() or heartbeat.custom_mode != mode:
            raise RuntimeError('Vzlet: ztráta telemetrie nebo armed GUIDED.')
        if pos_time == last_sample:
            continue
        last_sample = pos_time
        result = monitor.update(position.relative_alt/1000., -position.vz/100., pos_time, clock())
        if monitor.failure:
            raise RuntimeError(result.state)
        if result.ready:
            report('AIRBORNE: výška potvrzena a ustálena.')
            return
