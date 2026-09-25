"""Letový backend pro ArduPilot Copter přes MAVLink.

Převádí záměry MissionCommand na povely autopilota a skládá MissionInput
z telemetrie. Samotné rozhodování zůstává v MissionController.

Zásady:
- SET_GUIDED / ARM / TAKEOFF se posílají jednou pro každé request_id; výsledek
  přijde z COMMAND_ACK jako CommandFeedback. Nic se automaticky neopakuje.
- VELOCITY a BRAKE se posílají jen v GUIDED a armed. Mimo GUIDED má stroj pilot.
- Opuštění GUIDED po jeho potvrzení = ruční převzetí (manual_override, RELEASE).
- ArduPilot sám zastaví, pokud 3 s nepřijde rychlostní povel (výpadek Pi).
- Lokální souřadnice: LOCAL_POSITION_NED minus poloha zachycená na startu, takže
  (0, 0) je místo, kde dron stál při capture_origin().
"""
import math
import sys
import threading
import time
from .approach import VehicleState
from .config import StartReference
from .field import GroundPoint
from .mission_controller import CommandFeedback, MissionInput
from .telemetry import TelemetryStore

# Číselné hodnoty MAVLink, aby modul šel testovat bez pymavlink.
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128
MAV_RESULT_ACCEPTED = 0
MAV_RESULT_IN_PROGRESS = 5
MAV_FRAME_LOCAL_NED = 1
MAV_LANDED_STATE_ON_GROUND = 1
MAV_AUTOPILOT_ARDUPILOTMEGA = 3
VELOCITY_ONLY_MASK = 3527       # ignorovat polohu, zrychlení, yaw a yaw rate
VELOCITY_YAW_MASK = 2503        # rychlost + pevný kurz (yaw), ignorovat polohu, zrychlení, yaw rate
COPTER_MODES = {'STABILIZE': 0, 'ALT_HOLD': 2, 'AUTO': 3, 'GUIDED': 4, 'LOITER': 5,
                'RTL': 6, 'LAND': 9, 'BRAKE': 17}
COPTER_TYPES = (2, 3, 4, 13, 14, 15, 29, 35)
STREAMS = {33: 10, 30: 30, 24: 2, 32: 20, 245: 2}  # GPI, ATTITUDE, GPS_RAW, LOCAL_NED, EXT_SYS_STATE
MAX_SPEED = 2.
MAX_CLIMB = .7


class ArduPilotBackend:
    def __init__(self, connection, *, clock=time.monotonic, log=None, guided_mode=COPTER_MODES['GUIDED']):
        self.connection, self.clock, self.guided_mode = connection, clock, guided_mode
        self.log = log or (lambda text: print(text, file=sys.stderr))
        self.store = TelemetryStore()
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.system = getattr(connection, 'target_system', 1)
        self.component = getattr(connection, 'target_component', 1)
        self.heartbeat = None      # (čas, armed, custom_mode)
        self.local = None          # (čas, x, y, z, vx, vy, vz)
        self.global_pos = None     # (čas, lat, lon, relative_alt)
        self.landed_state = None
        self.origin = None         # (x, y, z) LOCAL_NED na startu
        self.start = None
        self.pending = {}          # MAV_CMD -> request_id
        self.feedback = []
        self.requested_commands = set()   # MAV_CMD, které už řadič vyžádal
        self.sent_ids = set()             # request_id už odeslané (každé jen jednou)
        self.guided_confirmed = False
        self.manual = False
        self.last_sent = None
        self.boot_ms = 0
        self.yaw = None            # rad, poslední ATTITUDE
        self.hold_yaw = None       # rad, kurz zachycený na startu; drží se po celý let
        self.land_sent = False

    # ---------------------------------------------------------------- příjem
    def ingest(self, message, now=None):
        now = self.clock() if now is None else now
        if (message.get_srcSystem(), message.get_srcComponent()) != (self.system, self.component):
            return
        kind = message.get_type()
        self.store.ingest(message, now)
        with self._lock:
            if kind == 'HEARTBEAT':
                armed = bool(message.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
                self.heartbeat = (now, armed, message.custom_mode)
                guided = message.custom_mode == self.guided_mode
                if guided and MAV_CMD_DO_SET_MODE in self.requested_commands:
                    self.guided_confirmed = True
                if self.guided_confirmed and not guided and not self.manual and not self.land_sent:
                    self.manual = True
                    self.log(f'MAVLink: režim {message.custom_mode} místo GUIDED -> převzetí pilotem.')
            elif kind == 'LOCAL_POSITION_NED':
                values = (message.x, message.y, message.z, message.vx, message.vy, message.vz)
                if all(math.isfinite(v) for v in values):
                    self.local = (now, *values)
                self.boot_ms = message.time_boot_ms
            elif kind == 'GLOBAL_POSITION_INT':
                self.global_pos = (now, message.lat/1e7, message.lon/1e7, message.relative_alt/1000)
            elif kind == 'ATTITUDE':
                if math.isfinite(message.yaw):
                    self.yaw = message.yaw
            elif kind == 'EXTENDED_SYS_STATE':
                self.landed_state = message.landed_state
            elif kind == 'COMMAND_ACK':
                request = self.pending.get(message.command)
                if request is not None and message.result != MAV_RESULT_IN_PROGRESS:
                    del self.pending[message.command]
                    self.feedback.append(CommandFeedback(request, message.result == MAV_RESULT_ACCEPTED))
                    if message.result != MAV_RESULT_ACCEPTED:
                        self.log(f'MAVLink: povel {message.command} odmítnut (result {message.result}).')
            elif kind == 'STATUSTEXT':
                self.log(f'ArduPilot: {message.text}')

    def snapshot(self):
        """Kompatibilní s MAVLinkTelemetry.snapshot() pro DetectionPipeline."""
        return self.store.snapshot()

    # ---------------------------------------------------------------- vlákno
    def start_io(self):
        for message_id, hz in STREAMS.items():
            self._command(MAV_CMD_SET_MESSAGE_INTERVAL, (message_id, int(1e6/hz), 0, 0, 0, 0, 0))
        self._thread = threading.Thread(target=self._run, daemon=True, name='mavlink-backend')
        self._thread.start()

    def _run(self):
        last_heartbeat = 0.
        while not self._stop.is_set():
            try:
                now = self.clock()
                if now-last_heartbeat >= 1:
                    with self._send_lock:
                        # Onboard controller (18), autopilot invalid (8), active (4).
                        self.connection.mav.heartbeat_send(18, 8, 0, 0, 4)
                    last_heartbeat = now
                message = self.connection.recv_match(blocking=True, timeout=.05)
                if message is not None:
                    self.ingest(message)
            except Exception as error:  # Chyba spojení se projeví stárnutím telemetrie.
                self.store.error = str(error)
                self.log(f'MAVLink: {error}')
                time.sleep(.1)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # ---------------------------------------------------------------- stav
    def wait_ready(self, timeout=10.):
        deadline = self.clock()+timeout
        while self.clock() < deadline:
            if self._fresh(self.clock()):
                return True
            time.sleep(.05)
        return False

    def _fresh(self, now):
        with self._lock:
            return (self.heartbeat is not None and now-self.heartbeat[0] <= 2 and
                    self.local is not None and now-self.local[0] <= .3 and
                    self.global_pos is not None and now-self.global_pos[0] <= .5)

    def capture_origin(self):
        """Uloží aktuální polohu jako start (0, 0). Jen nearmovaný stroj s čerstvými daty."""
        now = self.clock()
        if not self._fresh(now):
            raise RuntimeError('Start: chybí čerstvý HEARTBEAT, LOCAL_POSITION_NED nebo GLOBAL_POSITION_INT.')
        with self._lock:
            if self.heartbeat[1]:
                raise RuntimeError('Start: stroj je armovaný, referenci lze zachytit jen na zemi.')
            _, x, y, z, *_ = self.local
            _, lat, lon, _ = self.global_pos
            self.origin = (x, y, z)
            self.start = StartReference(lat, lon)
            # Stejný kurz po celý let: záběr kamery je vždy stejně natočený a
            # chyby montáže kamery se v rozdílu červená-zelená odečtou.
            self.hold_yaw = self.yaw
        return self.start

    def vehicle(self):
        """Poloha a rychlost v lokální mapě (sever, východ od startu) a výška nad startem."""
        with self._lock:
            if self.local is None or self.origin is None:
                return None, math.nan, math.nan
            at, x, y, z, vx, vy, vz = self.local
            x0, y0, z0 = self.origin
        return VehicleState(GroundPoint(x-x0, y-y0), vx, vy, at), -(z-z0), -vz

    def mission_input(self, field, *, camera_ready, camera_locked, target, stop_requested=False, green=None):
        vehicle, height, up = self.vehicle()
        if vehicle is None:
            vehicle = VehicleState(GroundPoint(math.nan, math.nan), math.nan, math.nan, math.nan)
        with self._lock:
            heartbeat = self.heartbeat or (math.nan, False, None)
            feedback = self.feedback.pop(0) if self.feedback else None
            landed = self.landed_state
            manual = self.manual
        on_ground = landed == MAV_LANDED_STATE_ON_GROUND if landed is not None else height < .3
        return MissionInput(vehicle, field, height, up, heartbeat[0],
                            guided=heartbeat[2] == self.guided_mode, armed=heartbeat[1],
                            on_ground=on_ground, camera_ready=camera_ready, camera_locked=camera_locked,
                            target=target, feedback=feedback, manual_override=manual,
                            stop_requested=stop_requested, green=green)

    # ---------------------------------------------------------------- povely
    def _command(self, command, params):
        with self._send_lock:
            self.connection.mav.command_long_send(self.system, self.component, command, 0, *params)

    def _request(self, command, params, request_id):
        if request_id is None or request_id in self.sent_ids:
            return
        with self._lock:
            self.pending[command] = request_id
            self.requested_commands.add(command)
            self.sent_ids.add(request_id)
        self._command(command, params)

    def _velocity(self, north, east, down):
        values = (north, east, down)
        if not all(math.isfinite(v) for v in values) or math.hypot(north, east) > MAX_SPEED+1e-6:
            raise ValueError('Neplatný nebo příliš rychlý povel.')
        mask, yaw = (VELOCITY_ONLY_MASK, 0.) if self.hold_yaw is None else (VELOCITY_YAW_MASK, self.hold_yaw)
        with self._send_lock:
            self.connection.mav.set_position_target_local_ned_send(
                self.boot_ms, self.system, self.component, MAV_FRAME_LOCAL_NED, mask,
                0, 0, 0, north, east, down, 0, 0, 0, yaw, 0)
        self.last_sent = (north, east, down)

    def execute(self, command):
        """Provede jeden záměr řadiče. Vrací popis toho, co se skutečně odeslalo."""
        with self._lock:
            heartbeat = self.heartbeat
        guided = heartbeat is not None and heartbeat[2] == self.guided_mode
        armed = heartbeat is not None and heartbeat[1]
        action = command.action
        if action == 'SET_GUIDED':
            self._request(MAV_CMD_DO_SET_MODE, (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, self.guided_mode,
                                                0, 0, 0, 0, 0), command.request_id)
            return 'SET_MODE_GUIDED'
        if action == 'ARM':
            self._request(MAV_CMD_COMPONENT_ARM_DISARM, (1, 0, 0, 0, 0, 0, 0), command.request_id)
            return 'ARM'
        if action == 'TAKEOFF':
            height = command.height_setpoint_m
            if height is None or not 1 <= height <= 20:
                raise ValueError('Neplatná výška vzletu.')
            self._request(MAV_CMD_NAV_TAKEOFF, (0, 0, 0, 0, 0, 0, height), command.request_id)
            return 'TAKEOFF'
        if action in ('VELOCITY', 'BRAKE'):
            if not (guided and armed):
                return 'NOT_SENT_NOT_GUIDED_ARMED'
            if action == 'BRAKE':
                self._velocity(0., 0., 0.)
                return 'ZERO_VELOCITY'
            _, height, _ = self.vehicle()
            down = 0.
            if command.height_setpoint_m is not None and math.isfinite(height):
                # Držení výšky: P regulátor, kladné NED vz = dolů.
                down = -max(-MAX_CLIMB, min(MAX_CLIMB, .8*(command.height_setpoint_m-height)))
            self._velocity(command.north_m_s, command.east_m_s, down)
            return 'VELOCITY'
        if action == 'LAND':
            if guided and armed and not self.land_sent:
                self.set_mode('LAND')
                self.land_sent = True
                return 'SET_MODE_LAND'
            return 'NONE'
        # WAIT, TAKEOFF_MONITOR (vzlet řídí autopilot), RELEASE, NONE, HOLD: nic neposílat.
        return 'NONE'

    def set_mode(self, name):
        """Jednorázová změna režimu při ukončení (LOITER/RTL/LAND); bez čekání na ACK."""
        mode = COPTER_MODES[name]
        self._command(MAV_CMD_DO_SET_MODE, (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode, 0, 0, 0, 0, 0))


def connect(endpoint, baud=115200, *, timeout=10., target_system=None):
    """Otevře MAVLink a počká na heartbeat ArduPilot Copteru."""
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection(endpoint, baud=baud, source_system=245, source_component=191)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        message = connection.recv_match(type='HEARTBEAT', blocking=True, timeout=.5)
        if message is None or message.autopilot != MAV_AUTOPILOT_ARDUPILOTMEGA:
            continue
        if target_system is not None and message.get_srcSystem() != target_system:
            continue
        if message.type not in COPTER_TYPES:
            connection.close()
            raise RuntimeError(f'Autopilot není Copter (MAV_TYPE {message.type}).')
        connection.target_system = message.get_srcSystem()
        connection.target_component = message.get_srcComponent()
        return connection
    connection.close()
    raise RuntimeError('Nepřišel heartbeat ArduPilotu. Zkontrolujte port, baudrate a SERIALx_PROTOCOL=2.')
