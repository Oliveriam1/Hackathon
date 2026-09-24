"""Dependency-injected mission controller; no concrete autopilot is assumed."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Callable, Protocol

import numpy as np

from .geolocation import geodetic_to_local, geolocate
from .models import CameraIntrinsics, CameraMount, GeoCoordinate, LocalPoint, MissionState, Point
from .navigation import FlightController, plan_mission
from .publisher import Publisher

logger = logging.getLogger(__name__)


class MissionCamera(Protocol):
    """read() must honor its deadline; a blocked device must raise TimeoutError."""
    def open(self) -> None: ...
    def read(self, timeout_s: float) -> np.ndarray: ...
    def close(self) -> None: ...


class CircleDetector(Protocol):
    def detect_circle(self, frame: np.ndarray) -> Point | None: ...


@dataclass(frozen=True, slots=True)
class MissionConfig:
    start: GeoCoordinate
    destination: GeoCoordinate
    corners: tuple[GeoCoordinate, ...]
    intrinsics: CameraIntrinsics
    camera_mount: CameraMount
    transit_altitude_agl_m: float
    max_altitude_agl_m: float
    scan_timeout_s: float = 60.0
    command_timeout_s: float = 30.0
    yaw_deg: float = 0.0
    coverage_margin: float = 1.1

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v <= 0 for v in (self.scan_timeout_s, self.command_timeout_s)):
            raise ValueError("Mission timeouts must be finite and positive")
        if self.camera_mount != CameraMount(0, -90, 0):
            raise ValueError("Automatic coverage planning currently requires mount (0,-90,0), nadir")


@dataclass(frozen=True, slots=True)
class MissionReport:
    state: MissionState
    coordinates: LocalPoint | None
    error: str | None
    history: tuple[MissionState, ...]


class MissionController:
    def __init__(self, config: MissionConfig, flight: FlightController, camera: MissionCamera,
                 detector: CircleDetector, publisher: Publisher, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 on_state: Callable[[MissionState], None] | None = None) -> None:
        self.config, self.flight, self.camera = config, flight, camera
        self.detector, self.publisher = detector, publisher
        self.clock, self.sleep = clock, sleep
        self.on_state = on_state
        self.history = [MissionState.INIT]
        self._used = False

    def _state(self, state: MissionState) -> None:
        self.history.append(state)
        logger.info("STATE: %s", state.value)
        if self.on_state is not None:
            # Observers only copy status; a display error must not alter flight.
            try:
                self.on_state(state)
            except Exception:
                logger.exception("Mission state observer failed")

    def run(self) -> MissionReport:
        """Run once. Every failure stops progression, aborts the adapter and closes camera."""
        if self._used:
            raise RuntimeError("Create a new controller for each mission")
        self._used = True
        coordinates = None
        error = None
        try:
            config, intrinsics = self.config, self.config.intrinsics
            # Use the narrower side around the optical center, also for off-center calibration.
            half_width = min(intrinsics.cx, intrinsics.width - intrinsics.cx)
            half_height = min(intrinsics.cy, intrinsics.height - intrinsics.cy)
            h_fov = math.degrees(2 * math.atan(half_width / intrinsics.fx))
            v_fov = math.degrees(2 * math.atan(half_height / intrinsics.fy))
            # Validate the whole geometric plan before any takeoff/motion intent.
            plan = plan_mission(config.start, config.destination, config.corners, h_fov, v_fov,
                                config.transit_altitude_agl_m, config.max_altitude_agl_m,
                                yaw_deg=config.yaw_deg, coverage_margin=config.coverage_margin)
            initial = self.flight.get_telemetry()
            offset = geodetic_to_local(config.start, initial.position)
            if math.hypot(offset.x, offset.y) > 2 or initial.altitude_agl_m > 0.5:
                raise ValueError("Mission must start on the ground within 2 m of point A")
            self.camera.open()  # Catch unavailable camera before requesting takeoff.
            self._state(MissionState.MOVING_TO_B)
            for command in plan.commands[:2]:
                self.flight.execute(command, config.command_timeout_s)
            self._state(MissionState.CALCULATING_CENTER)
            logger.info("Scan center: %s, required AGL: %.2f m", plan.center, plan.scan_altitude_agl_m)
            self.flight.execute(plan.commands[2], config.command_timeout_s)
            self._state(MissionState.ASCENDING)
            self.flight.execute(plan.commands[3], config.command_timeout_s)
            telemetry = self.flight.get_telemetry()
            position_error = geodetic_to_local(plan.center, telemetry.position)
            yaw_error = (telemetry.attitude.yaw_deg - config.yaw_deg + 180) % 360 - 180
            if (math.hypot(position_error.x, position_error.y) > 2
                    or abs(telemetry.altitude_agl_m - plan.scan_altitude_agl_m) > 1
                    or abs(telemetry.attitude.roll_deg) > 2 or abs(telemetry.attitude.pitch_deg) > 2
                    or abs(yaw_error) > 2):
                raise RuntimeError("Scan pose was not confirmed by telemetry")
            self._state(MissionState.SCANNING)
            deadline = self.clock() + config.scan_timeout_s
            while True:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError(f"No circle found within {config.scan_timeout_s:g} seconds")
                frame = self.camera.read(timeout_s=remaining)
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    raise RuntimeError("Camera returned no valid frame")
                if frame.shape[:2] != (intrinsics.height, intrinsics.width):
                    raise ValueError("Frame resolution does not match camera calibration")
                pixel = self.detector.detect_circle(frame)
                if self.clock() >= deadline:
                    raise TimeoutError("Scan deadline exceeded")
                if pixel is None:
                    self.sleep(min(0.02, max(0, deadline - self.clock())))
                    continue
                self._state(MissionState.TARGET_FOUND)
                telemetry = self.flight.get_telemetry()
                location = geolocate(pixel, telemetry, intrinsics, config.camera_mount)
                coordinates = geodetic_to_local(config.start, location.coordinate)
                self._state(MissionState.SENDING_DATA)
                self.publisher.publish(coordinates)
                self._state(MissionState.COMPLETE)
                break
        except Exception as exc:
            error = str(exc)
            logger.exception("Mission failed")
            self._state(MissionState.FAILED)
            try:
                self.flight.abort()
            except Exception:
                logger.exception("Flight adapter abort failed")
        except KeyboardInterrupt:
            self._state(MissionState.FAILED)
            self.flight.abort()
            raise
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                logger.exception("Camera cleanup failed")
                error = error or str(exc)
                if self.history[-1] != MissionState.FAILED:
                    self._state(MissionState.FAILED)
        return MissionReport(self.history[-1], coordinates, error, tuple(self.history))
