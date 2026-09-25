"""Test dronu na stole BEZ VRTULÍ: spojení, stav autopilota a roztočení motorů.

  python3 tools/bench_test.py --connect /dev/serial0 --baud 921600              # jen stav, nic se netočí
  python3 tools/bench_test.py --connect /dev/serial0 --baud 921600 --motor-test # každý motor zvlášť 2 s
  python3 tools/bench_test.py --connect /dev/serial0 --baud 921600 --spin 5     # armovat, 5 s volnoběh, odzbrojit
  python3 tools/bench_test.py --connect /dev/serial0 --baud 921600 --takeoff-test
      # stejné povely jako mise: GUIDED -> ARM -> TAKEOFF 1 m, 3 s, pak LAND a odzbrojení

Každý test s motory vyžaduje napsat BEZ VRTULI. Ctrl+C kdykoliv = okamžité odzbrojení.
"""
import argparse
import math
import time

MODES = {0: 'STABILIZE', 2: 'ALT_HOLD', 3: 'AUTO', 4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 9: 'LAND', 17: 'BRAKE'}
GPS_FIX = {0: 'žádná GPS', 1: 'bez fixu', 2: '2D', 3: '3D', 4: 'DGPS', 5: 'RTK float', 6: 'RTK fixed'}
PARAMS = ('ARMING_CHECK', 'FENCE_ENABLE', 'FENCE_TYPE', 'FENCE_ACTION', 'FENCE_ALT_MAX', 'FS_THR_ENABLE',
          'BATT_FS_LOW_ACT', 'MOT_SPIN_ARM', 'MOT_SPIN_MIN', 'SERIAL1_PROTOCOL', 'SERIAL2_PROTOCOL',
          'SERIAL2_BAUD', 'FRAME_CLASS', 'FRAME_TYPE', 'DISARM_DELAY')


class Bench:
    def __init__(self, connection, mavlink):
        self.c, self.m = connection, mavlink
        self.hb = None
        self.last = {}
        self.texts = []

    def pump(self, seconds=.1):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            msg = self.c.recv_match(blocking=True, timeout=.05)
            if msg is None:
                continue
            if msg.get_srcSystem() != self.c.target_system:
                continue
            kind = msg.get_type()
            self.last[kind] = msg
            if kind == 'HEARTBEAT' and msg.get_srcComponent() == self.c.target_component:
                self.hb = msg
            elif kind == 'STATUSTEXT':
                self.texts.append(msg.text)
                print(f'  ArduPilot: {msg.text}')

    @property
    def armed(self):
        return bool(self.hb and self.hb.base_mode & self.m.MAV_MODE_FLAG_SAFETY_ARMED)

    @property
    def mode(self):
        return MODES.get(self.hb.custom_mode, str(self.hb.custom_mode)) if self.hb else '?'

    def command(self, command, *params, wait=5., label=''):
        params = list(params)+[0]*(7-len(params))
        self.c.mav.command_long_send(self.c.target_system, self.c.target_component, command, 0, *params)
        end = time.monotonic()+wait
        while time.monotonic() < end:
            msg = self.c.recv_match(type=['COMMAND_ACK', 'STATUSTEXT', 'HEARTBEAT'], blocking=True, timeout=.1)
            if msg is None:
                continue
            if msg.get_type() == 'STATUSTEXT':
                print(f'  ArduPilot: {msg.text}')
                self.texts.append(msg.text)
            elif msg.get_type() == 'HEARTBEAT' and msg.get_srcSystem() == self.c.target_system \
                    and msg.get_srcComponent() == self.c.target_component:
                self.hb = msg
            elif msg.get_type() == 'COMMAND_ACK' and msg.command == command:
                if msg.result == self.m.MAV_RESULT_IN_PROGRESS:
                    continue
                ok = msg.result == self.m.MAV_RESULT_ACCEPTED
                print(f'  {label or command}: {"PŘIJATO" if ok else f"ODMÍTNUTO (result {msg.result})"}')
                return ok
        print(f'  {label or command}: bez odpovědi (timeout)')
        return False

    def set_mode(self, name):
        mode = self.c.mode_mapping()[name]
        return self.command(self.m.MAV_CMD_DO_SET_MODE, self.m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode,
                            label=f'režim {name}')

    def arm(self, value=1):
        return self.command(self.m.MAV_CMD_COMPONENT_ARM_DISARM, value, label='ARM' if value else 'DISARM')

    def disarm_now(self):
        """Odzbrojení; když normální nejde (stroj ve vzduchu podle autopilota), vynucené."""
        if not self.arm(0):
            self.command(self.m.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196, label='DISARM (vynuceně)')


def status(bench):
    m = bench.m
    bench.c.mav.command_long_send(bench.c.target_system, bench.c.target_component,
                                  m.MAV_CMD_REQUEST_MESSAGE, 0, 148, 0, 0, 0, 0, 0, 0)  # AUTOPILOT_VERSION
    for message_id, hz in ((1, 2), (24, 2), (33, 5), (30, 5), (193, 2)):  # SYS_STATUS, GPS, POS, ATT, EKF
        bench.c.mav.command_long_send(bench.c.target_system, bench.c.target_component,
                                      m.MAV_CMD_SET_MESSAGE_INTERVAL, 0, message_id, int(1e6/hz), 0, 0, 0, 0, 0)
    bench.pump(3)
    print('\n=== STAV AUTOPILOTA ===')
    version = bench.last.get('AUTOPILOT_VERSION')
    if version:
        v = version.flight_sw_version
        print(f'Firmware: {v >> 24 & 255}.{v >> 16 & 255}.{v >> 8 & 255}')
    print(f'Režim: {bench.mode}, armed: {bench.armed}, MAV_TYPE: {bench.hb.type} (2 = quad)')
    sys_status = bench.last.get('SYS_STATUS')
    if sys_status:
        print(f'Baterie: {sys_status.voltage_battery/1000:.2f} V')
    gps = bench.last.get('GPS_RAW_INT')
    if gps:
        print(f'GPS: {GPS_FIX.get(gps.fix_type, gps.fix_type)}, satelitů {gps.satellites_visible}, '
              f'HDOP {gps.eph/100 if gps.eph != 65535 else "?"}')
    else:
        print('GPS: žádná zpráva GPS_RAW_INT')
    ekf = bench.last.get('EKF_STATUS_REPORT')
    if ekf:
        flags = ekf.flags
        print(f'EKF: horizontální poloha {"OK" if flags & 16 else "NE"}, '
              f'výška {"OK" if flags & 32 else "NE"} (bez GPS venku je NE normální)')
    att = bench.last.get('ATTITUDE')
    if att:
        print(f'Náklon: roll {math.degrees(att.roll):.1f}°, pitch {math.degrees(att.pitch):.1f}°, '
              f'kurz {math.degrees(att.yaw) % 360:.0f}° (natočte dron a sledujte změnu)')
    pos = bench.last.get('GLOBAL_POSITION_INT')
    if pos:
        print(f'Výška nad home: {pos.relative_alt/1000:.2f} m')
    print('\n=== PARAMETRY ===')
    for name in PARAMS:
        bench.c.mav.param_request_read_send(bench.c.target_system, bench.c.target_component, name.encode(), -1)
        msg = bench.c.recv_match(type='PARAM_VALUE', blocking=True, timeout=1.5)
        while msg is not None and msg.param_id.rstrip('\x00') != name:
            msg = bench.c.recv_match(type='PARAM_VALUE', blocking=True, timeout=1.5)
        print(f'{name:18s} {msg.param_value:g}' if msg else f'{name:18s} (neexistuje / bez odpovědi)')
    print('\n=== PRE-ARM KONTROLA ===')
    bench.texts.clear()
    bench.command(m.MAV_CMD_RUN_PREARM_CHECKS, wait=3., label='kontrola')
    bench.pump(2)
    problems = [t for t in bench.texts if 'PreArm' in t]
    print('Pre-arm: bez chyb' if not problems else 'Pre-arm chyby výše – dokud tam jsou, armování nepůjde.')


def confirm():
    answer = input('Jsou SUNDANÉ VŠECHNY VRTULE? Napište BEZ VRTULI: ').strip().upper()
    if answer not in ('BEZ VRTULI', 'BEZ VRTULÍ'):
        raise SystemExit('Zrušeno.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--connect', required=True)
    parser.add_argument('--baud', type=int, default=115200)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--motor-test', action='store_true', help='Každý motor 2 s na 8 % (i bez armování).')
    group.add_argument('--spin', type=float, metavar='S', help='Armovat, S sekund volnoběh, odzbrojit.')
    group.add_argument('--takeoff-test', action='store_true', help='GUIDED + ARM + TAKEOFF 1 m na 3 s, pak LAND.')
    args = parser.parse_args()
    from pymavlink import mavutil
    print(f'Připojuji {args.connect} @ {args.baud} ...')
    c = mavutil.mavlink_connection(args.connect, baud=args.baud, source_system=245, source_component=191)
    hb = None
    end = time.monotonic()+10
    while time.monotonic() < end:
        msg = c.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg is not None and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            hb = msg
            break
    if hb is None:
        raise SystemExit('Žádný heartbeat ArduPilotu. Zkontrolujte kabel TX/RX, --baud, SERIALx_PROTOCOL=2 '
                         'a že nic jiného (Mission Planner přes stejný port) port neblokuje.')
    c.target_system, c.target_component = hb.get_srcSystem(), hb.get_srcComponent()
    print(f'Spojeno: system {c.target_system}, component {c.target_component}')
    bench = Bench(c, mavutil.mavlink)
    bench.hb = hb
    status(bench)
    if not (args.motor_test or args.spin or args.takeoff_test):
        return
    print()
    confirm()
    m = mavutil.mavlink
    try:
        if args.motor_test:
            print('Motor test: motory v pořadí A, B, C, D (A = vpravo vpředu, dál po směru hodinových ručiček).')
            for motor in range(1, 5):
                print(f'  motor {"ABCD"[motor-1]} ...')
                # param3 = 8 % plynu, param4 = 2 s, param5 = 0 jen tento motor, param6 = 1 pořadí sekvence
                bench.command(m.MAV_CMD_DO_MOTOR_TEST, motor, 0, 8, 2, 0, 1, label=f'motor {"ABCD"[motor-1]}')
                bench.pump(2.5)
            print('Zkontrolujte: každý motor se točil na správném místě a správným směrem (viz ArduPilot Motor order).')
        elif args.spin:
            gps = bench.last.get('GPS_RAW_INT')
            mode = 'GUIDED' if gps and gps.fix_type >= 3 else 'STABILIZE'
            print(f'Režim {mode} ({"GPS fix" if mode == "GUIDED" else "bez GPS fixu – test jen volnoběhu"})')
            if bench.set_mode(mode) and bench.arm(1):
                print(f'Motory by se měly točit volnoběhem {args.spin:.0f} s ...')
                bench.pump(args.spin)
            bench.disarm_now()
        else:
            print('Test povelů mise: GUIDED -> ARM -> TAKEOFF 1 m (motory se roztočí víc) -> 3 s -> LAND -> DISARM')
            if bench.set_mode('GUIDED') and bench.arm(1) and \
                    bench.command(m.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 1, label='TAKEOFF 1 m'):
                bench.pump(3)
            bench.set_mode('LAND')
            bench.pump(1)
            bench.disarm_now()
    except KeyboardInterrupt:
        print('\nCtrl+C: odzbrojuji.')
        bench.disarm_now()
    bench.pump(1.5)
    print(f'Konec: armed = {bench.armed}, režim {bench.mode}')


if __name__ == '__main__':
    main()
