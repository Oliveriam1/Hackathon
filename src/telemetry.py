"""Read-only telemetry interfaces, independent of any autopilot protocol."""

from dataclasses import dataclass
from typing import Protocol

from .models import UAVTelemetry


class TelemetryProvider(Protocol):
    """Supply validated telemetry corresponding to the frame exposure time.

    Adapters must convert native axes/units to FRD/NED degrees and true-North
    yaw, provide explicit AGL, and reject unavailable/stale measurements.
    A real adapter must handle frame/telemetry time alignment.
    """

    def get_telemetry(self) -> UAVTelemetry:
        """Return a valid sample, or raise an exception when unavailable."""
        ...


@dataclass(frozen=True, slots=True)
class StaticTelemetryProvider:
    """Explicit, unchanging test data; never a source of live UAV telemetry."""

    telemetry: UAVTelemetry

    def get_telemetry(self) -> UAVTelemetry:
        return self.telemetry
