"""Local-coordinate JSON publishing. UDP sending never happens on import."""

import json
import math
import socket
from typing import Protocol

from .models import LocalPoint


def target_payload(coordinates: LocalPoint) -> dict:
    """x East / y North in meters from the mission's start point A."""
    return {"target_found": True, "coordinates": {"x": coordinates.x, "y": coordinates.y}}


class Publisher(Protocol):
    def publish(self, coordinates: LocalPoint) -> None: ...


class ConsolePublisher:
    def publish(self, coordinates: LocalPoint) -> None:
        print(json.dumps(target_payload(coordinates), allow_nan=False), flush=True)


class UDPPublisher:
    """One datagram to an explicit PC endpoint; UDP does not acknowledge delivery."""
    def __init__(self, host: str, port: int, timeout_s: float = 2.0) -> None:
        if not host or not 1 <= port <= 65535 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid UDP endpoint or timeout")
        self.host, self.port, self.timeout_s = host, port, timeout_s

    def publish(self, coordinates: LocalPoint) -> None:
        data = json.dumps(target_payload(coordinates), allow_nan=False).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(self.timeout_s)
            sent = client.sendto(data, (self.host, self.port))
        if sent != len(data):
            raise OSError("Incomplete UDP datagram")
