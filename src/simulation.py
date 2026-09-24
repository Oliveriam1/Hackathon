"""Explicit simulation adapters. They never connect to a UAV or open a camera."""

import logging
import numpy as np

from .models import Attitude, FlightCommand, GeoCoordinate, UAVTelemetry

logger = logging.getLogger(__name__)


class SimulatedFlightController:
    """Instantly complete recorded intents; no claim of real flight physics."""
    def __init__(self, start: GeoCoordinate) -> None:
        self.telemetry = UAVTelemetry(start, 0.0, Attitude(0, 0, 0))
        self.commands: list[FlightCommand] = []
        self.aborted = False

    def get_telemetry(self) -> UAVTelemetry:
        return self.telemetry

    def execute(self, command: FlightCommand, timeout_s: float) -> None:
        if self.aborted or timeout_s <= 0:
            raise RuntimeError("Simulated flight is aborted or command timed out")
        logger.info("SIMULATED %s -> %s at AGL %.2f m", command.action.value,
                    command.position, command.altitude_agl_m)
        self.commands.append(command)
        self.telemetry = UAVTelemetry(command.position, command.altitude_agl_m,
                                      Attitude(0, 0, command.yaw_deg))

    def abort(self) -> None:
        self.aborted = True


class StaticImageCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def read(self, timeout_s: float) -> np.ndarray:
        if not self.opened or timeout_s <= 0:
            raise RuntimeError("Simulated camera unavailable")
        return self.frame.copy()

    def close(self) -> None:
        self.opened = False
