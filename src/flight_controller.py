"""Rozhraní budoucího letového adaptéru pro simulaci a autopilot.

Neexistuje zde konkrétní adaptér ani automatické armování. Telemetrie je
oddělená v telemetry.py; fáze 1 tyto metody nikde nevolá.
"""
from typing import Protocol
from .field import GroundPoint


class FlightController(Protocol):
    def takeoff(self, height_m: float) -> None:
        """Požádá o vzlet; dokončení musí mise ověřit z telemetrie."""
        ...

    def move_to(self, target: GroundPoint, height_m: float) -> None:
        """Bod v lokální mapě; převod do souřadnic autopilota patří adaptéru."""
        ...

    def hold(self) -> None:
        """Požádá o zastavení a držení polohy."""
        ...
