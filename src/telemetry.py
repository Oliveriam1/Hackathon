"""Protocol-neutral read-only telemetry; no real autopilot adapter yet."""

from dataclasses import dataclass
from typing import Protocol

from .models import Attitude, GeoCoordinate, UAVTelemetry


class TelemetryProvider(Protocol):
    """Return valid telemetry with explicit AGL, true-North yaw and FRD/NED axes."""

    def get_telemetry(self) -> UAVTelemetry: ...


@dataclass(frozen=True, slots=True)
class StaticTelemetryProvider:
    """Explicit test/replay sample, never an implicit production default."""

    telemetry: UAVTelemetry

    def get_telemetry(self) -> UAVTelemetry:
        return self.telemetry


def telemetry_from_dict(data: dict) -> UAVTelemetry:
    """Decode an explicitly supplied snapshot, without inferring AGL."""
    return UAVTelemetry(
        position=GeoCoordinate(**data["position"]),
        altitude_agl_m=data["altitude_agl_m"],
        attitude=Attitude(**data["attitude"]),
        timestamp=data.get("timestamp"),
    )
