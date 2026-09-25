"""Rozhraní budoucího regulátoru kamery. Hardware zůstává v gimbal.py."""
from typing import Protocol
from .geolocation import GimbalAngles
from .vision import Observation


class GimbalController(Protocol):
    def update(self, observation: Observation, angles: GimbalAngles,
               *, sample_time: float, now: float) -> GimbalAngles | None:
        """Požadované úhly ve stupních, nebo None bez platné korekce. Bez zápisu na GPIO."""
        ...
