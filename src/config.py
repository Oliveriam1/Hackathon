"""Explicit mission configuration; no real coordinates or FOV guessed at runtime."""

import json
from pathlib import Path

from .mission import MissionConfig
from .models import CameraIntrinsics, CameraMount, GeoCoordinate


def load_mission_config(path: Path) -> MissionConfig:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    camera = data["camera"]
    intrinsics = CameraIntrinsics.from_fov(camera["width"], camera["height"],
                                           camera["horizontal_fov_deg"], camera["vertical_fov_deg"])
    return MissionConfig(
        start=GeoCoordinate(**data["start"]), destination=GeoCoordinate(**data["destination"]),
        corners=tuple(GeoCoordinate(**point) for point in data["corners"]),
        intrinsics=intrinsics, camera_mount=CameraMount(**camera["mount"]),
        transit_altitude_agl_m=data["transit_altitude_agl_m"],
        max_altitude_agl_m=data["max_altitude_agl_m"],
        scan_timeout_s=data.get("scan_timeout_s", 60.0),
        command_timeout_s=data.get("command_timeout_s", 30.0),
        yaw_deg=data.get("yaw_deg", 0.0), coverage_margin=data.get("coverage_margin", 1.1),
    )
