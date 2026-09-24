"""MAVLink telemetry worker. Requests streams; never sends flight commands."""
import math
import threading
import time


MESSAGE_IDS = {'GLOBAL_POSITION_INT': (33, 10), 'ATTITUDE': (30, 30), 'GPS_RAW_INT': (24, 2)}
MAX_AGE = {'HEARTBEAT': 3.0, 'GLOBAL_POSITION_INT': 0.5, 'ATTITUDE': 0.2, 'GPS_RAW_INT': 2.0}


class TelemetryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = {}
        self.error = None

    def ingest(self, message, now):
        kind = message.get_type()
        if kind == 'HEARTBEAT':
            values = {'armed': bool(message.base_mode & 128), 'custom_mode': message.custom_mode,
                      'vehicle_type': message.type, 'system_status': message.system_status}
        elif kind == 'GLOBAL_POSITION_INT':
            values = {'latitude_deg': message.lat/1e7, 'longitude_deg': message.lon/1e7,
                      'relative_alt_m': message.relative_alt/1000,
                      'velocity_ned_m_s': [message.vx/100, message.vy/100, message.vz/100],
                      'time_boot_ms': message.time_boot_ms}
        elif kind == 'ATTITUDE':
            values = {'roll_deg': math.degrees(message.roll), 'pitch_deg': math.degrees(message.pitch),
                      'yaw_deg': math.degrees(message.yaw), 'time_boot_ms': message.time_boot_ms}
        elif kind == 'GPS_RAW_INT':
            values = {'fix_type': message.fix_type,
                      'satellites_visible': None if message.satellites_visible == 255 else message.satellites_visible,
                      'hdop': None if message.eph == 65535 else message.eph/100}
        else:
            return
        # Reject NaN/Inf from a faulty telemetry source rather than emitting invalid JSON.
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        if not finite(values):
            return
        with self._lock:
            self._values[kind] = (now, values)

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            messages = {name: {'age_s': max(0, now-at), 'values': dict(values)}
                        for name, (at, values) in self._values.items()}
        missing = [name for name in MAX_AGE if name not in messages]
        stale = [name for name, item in messages.items() if item['age_s'] > MAX_AGE[name]]
        gps_ok = messages.get('GPS_RAW_INT', {}).get('values', {}).get('fix_type', 0) >= 3
        return {'status': 'error' if self.error else ('missing' if missing else ('stale' if stale else 'fresh')),
                'messages': messages, 'missing': missing, 'stale': stale, 'gps_3d_fix': gps_ok,
                'error': self.error, 'flight_ready': False,
                'time_basis': 'host_receive_monotonic_not_exposure_synchronized'}


class MAVLinkTelemetry:
    def __init__(self, endpoint, baud=115200, target_system=None):
        if baud <= 0 or (target_system is not None and not 1 <= target_system <= 255):
            raise ValueError('Neplatná rychlost nebo MAVLink system ID.')
        self.endpoint, self.baud, self.target_system = endpoint, baud, target_system
        self.store = TelemetryStore()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def __enter__(self):
        try:
            from pymavlink import mavutil
        except ImportError as error:
            raise RuntimeError('Chybí pymavlink. Nainstalujte jej do prostředí používaného programem.') from error
        self._thread = threading.Thread(target=self._run, args=(mavutil,), daemon=True, name='mavlink-telemetry')
        self._thread.start()
        if not self._ready.wait(7) or self.store.error:
            self.close()
            raise RuntimeError(self.store.error or 'MAVLink: nebyl přijat heartbeat autopilota.')
        return self

    def _run(self, mavutil):
        connection = None
        try:
            connection = mavutil.mavlink_connection(self.endpoint, baud=self.baud,
                                                    source_system=245, source_component=191)
            deadline = time.monotonic()+5
            system = component = None
            last_heartbeat = 0.0
            while not self._stop.is_set():
                now = time.monotonic()
                if now-last_heartbeat >= 1:
                    connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                                  mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                                                  mavutil.mavlink.MAV_STATE_ACTIVE)
                    last_heartbeat = now
                message = connection.recv_match(blocking=True, timeout=0.1)
                if system is None and now > deadline:
                    raise TimeoutError('MAVLink: heartbeat ArduPilotu nepřišel do 5 s.')
                if message is None:
                    continue
                if system is None:
                    if (message.get_type() != 'HEARTBEAT'
                            or message.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
                            or (self.target_system is not None and message.get_srcSystem() != self.target_system)):
                        continue
                    system, component = message.get_srcSystem(), message.get_srcComponent()
                    for message_id, hz in MESSAGE_IDS.values():
                        connection.mav.command_long_send(system, component,
                            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                            message_id, int(1e6/hz), 0, 0, 0, 0, 0)
                    self._ready.set()
                if message.get_srcSystem() == system and message.get_srcComponent() == component:
                    self.store.ingest(message, time.monotonic())
        except Exception as error:
            self.store.error = str(error)
            self._ready.set()
        finally:
            if connection is not None:
                connection.close()

    def snapshot(self):
        return self.store.snapshot()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError('MAVLink vlákno se nepodařilo zastavit.')

    def __exit__(self, *args):
        self.close()
