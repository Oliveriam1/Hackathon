"""Mission geometry and high-level command planning; no autopilot dependency."""

from dataclasses import dataclass
import math
from typing import Protocol

from .geolocation import geodetic_to_local, local_to_geodetic
from .models import FlightAction, FlightCommand, GeoCoordinate, LocalPoint, UAVTelemetry


class FlightController(Protocol):
    """Adapter contract: enforce timeouts and report real, completed motion.

    abort() must stop the mission using the platform's configured contingency;
    no universal land/hover/return command is assumed here.
    For live monitoring, get_telemetry() must return a thread-safe cached sample
    promptly, even while execute() waits for completion. The adapter owns all
    serial/protocol I/O; the video server does not read the flight link directly.
    """
    def get_telemetry(self) -> UAVTelemetry: ...
    def execute(self, command: FlightCommand, timeout_s: float) -> None: ...
    def abort(self) -> None: ...


def _validate_quad(corners: tuple[LocalPoint, ...]) -> None:
    if len(corners) != 4:
        raise ValueError("Four perimeter-ordered corners are required")
    turns = []
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        turns.append((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x))
    if not (all(v > 1e-9 for v in turns) or all(v < -1e-9 for v in turns)):
        raise ValueError("Corners must form a nondegenerate convex quadrilateral")


def polygon_centroid(corners: tuple[LocalPoint, ...]) -> LocalPoint:
    """Area centroid via the shoelace formula, not a simple vertex average."""
    _validate_quad(corners)
    cross = [a.x * b.y - b.x * a.y for a, b in zip(corners, corners[1:] + corners[:1])]
    area_twice = sum(cross)
    x = sum((a.x + b.x) * c for a, b, c in zip(corners, corners[1:] + corners[:1], cross))
    y = sum((a.y + b.y) * c for a, b, c in zip(corners, corners[1:] + corners[:1], cross))
    return LocalPoint(x / (3 * area_twice), y / (3 * area_twice))


def required_scan_altitude(corners: tuple[LocalPoint, ...], horizontal_fov_deg: float,
                           vertical_fov_deg: float, *, yaw_deg: float = 0,
                           coverage_margin: float = 1.1) -> float:
    """AGL covering all corners from the centroid with a level, nadir camera.

    Uses both FOV axes and the most distant corner on each side. Positive yaw
    rotates camera top from North toward East. Dimensions are meters, not GPS degrees.
    """
    center = polygon_centroid(corners)
    if not (0 < horizontal_fov_deg < 180 and 0 < vertical_fov_deg < 180):
        raise ValueError("FOV angles must be between 0 and 180 degrees")
    if not math.isfinite(coverage_margin) or coverage_margin < 1 or not math.isfinite(yaw_deg):
        raise ValueError("Invalid coverage margin or yaw")
    yaw = math.radians(yaw_deg)
    right = [abs((p.x - center.x) * math.cos(yaw) - (p.y - center.y) * math.sin(yaw)) for p in corners]
    forward = [abs((p.x - center.x) * math.sin(yaw) + (p.y - center.y) * math.cos(yaw)) for p in corners]
    return coverage_margin * max(max(right) / math.tan(math.radians(horizontal_fov_deg) / 2),
                                 max(forward) / math.tan(math.radians(vertical_fov_deg) / 2))


@dataclass(frozen=True, slots=True)
class MissionPlan:
    center: GeoCoordinate
    scan_altitude_agl_m: float
    commands: tuple[FlightCommand, ...]


def plan_mission(start: GeoCoordinate, destination: GeoCoordinate,
                 corners: tuple[GeoCoordinate, ...], horizontal_fov_deg: float,
                 vertical_fov_deg: float, transit_altitude_agl_m: float,
                 max_altitude_agl_m: float, *, yaw_deg: float = 0,
                 coverage_margin: float = 1.1) -> MissionPlan:
    """Generate takeoff, travel to B, travel to center and climb intents.

    No obstacle, terrain, radio-link or airspace model is available. This is a
    geometric plan; an actual flight adapter/operator must validate the route.
    """
    if not (math.isfinite(transit_altitude_agl_m) and math.isfinite(max_altitude_agl_m)
            and 0 < transit_altitude_agl_m <= max_altitude_agl_m):
        raise ValueError("Require 0 < transit AGL <= maximum AGL")
    geodetic_to_local(start, destination)  # Validate local operating range.
    local = tuple(geodetic_to_local(start, point) for point in corners)
    center = local_to_geodetic(start, polygon_centroid(local))
    height = max(transit_altitude_agl_m, required_scan_altitude(
        local, horizontal_fov_deg, vertical_fov_deg, yaw_deg=yaw_deg, coverage_margin=coverage_margin))
    if height > max_altitude_agl_m:
        raise ValueError("Required coverage altitude exceeds configured maximum AGL")
    return MissionPlan(center, height, (
        FlightCommand(FlightAction.TAKEOFF, start, transit_altitude_agl_m, yaw_deg),
        FlightCommand(FlightAction.GOTO, destination, transit_altitude_agl_m, yaw_deg),
        FlightCommand(FlightAction.GOTO, center, transit_altitude_agl_m, yaw_deg),
        FlightCommand(FlightAction.ASCEND, center, height, yaw_deg),
    ))
