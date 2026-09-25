"""Odesílání telemetrie míření přes síť (UDP, JSON) – např. přes WiFi na notebook.

Každá zpráva je jeden JSON objekt v jednom UDP paketu (UTF-8). Posílá se
nejvýše `rate_hz`-krát za sekundu, výsledek (úhel/souřadnice) vždy hned.
Cíl: 'IP:PORT' (např. 192.168.1.20:5005), pro všechny v síti '255.255.255.255:5005'.
Příjem na notebooku: python3 tools/receive_telemetry.py --port 5005

Zprávy:
  {"type": "aim", ...}     stav míření v tomto snímku
  {"type": "result", ...}  zprůměrovaný úhel na tečku (+ souřadnice, když je známa poloha dronu)
UDP nečeká na příjemce: když notebook neposlouchá, program běží dál beze změny.
"""
import json
import math
import socket
import time

SCHEMA_VERSION = 1


def parse_target(text):
    host, _, port = text.rpartition(':')
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f'Cíl telemetrie musí být IP:PORT, ne {text!r}.')
    return host, int(port)


def _clean(value):
    """NaN/inf nejsou platný JSON: nahradit None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class TelemetrySender:
    def __init__(self, target, *, rate_hz=10., clock=time.monotonic, sock=None):
        self.address = parse_target(target) if isinstance(target, str) else target
        self.interval = 1/rate_hz if rate_hz > 0 else 0.
        self.clock = clock
        self.sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.address[0].endswith('.255'):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.last_aim = None
        self.sequence = 0
        self.errors = 0

    def _send(self, message):
        self.sequence += 1
        message = {'schema_version': SCHEMA_VERSION, 'seq': self.sequence, 'time_unix': time.time(), **message}
        data = json.dumps(_clean(message), ensure_ascii=False, allow_nan=False).encode('utf-8')
        try:
            self.sock.sendto(data, self.address)
            return True
        except OSError:
            self.errors += 1  # síť dočasně nedostupná: míření kvůli tomu nezastavujeme
            return False

    def send_aim(self, aim_record):
        now = self.clock()
        if self.last_aim is not None and now-self.last_aim < self.interval:
            return False
        self.last_aim = now
        return self._send({'type': 'aim', **aim_record})

    def send_result(self, result_record):
        return self._send({'type': 'result', **result_record})

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
