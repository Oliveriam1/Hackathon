"""Local MJPEG monitor with bounded, latest-frame buffers and no hardware imports.

Capture, JPEG encoding, telemetry polling and HTTP clients run independently.
Slow clients skip frames; no camera, detector or flight call waits for a viewer.
All timestamps here use time.monotonic(), not GPS/UTC time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import math
import threading
import time
from typing import Callable

import cv2
import numpy as np

from .models import MissionState, UAVTelemetry
from .target_detector import DetectionResult, annotate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VideoSnapshot:
    frame: np.ndarray | None
    sequence: int
    captured_at: float
    state: MissionState
    telemetry: UAVTelemetry | None
    telemetry_at: float
    detection: DetectionResult | None
    detection_at: float
    camera_status: str
    detection_sequence: int = 0
    detection_shape: tuple[int, int] | None = None
    detection_mode: str = "WAITING"


def draw_overlay(snapshot: VideoSnapshot, *, now: float | None = None) -> np.ndarray:
    """Copy the frame; label stale/missing data instead of presenting it as live.

    Detection age is measured from its source frame, not completion of analysis.
    Geometry expires after one second because the camera/scene may have moved.
    """
    now = time.monotonic() if now is None else now
    frame = snapshot.frame
    if frame is None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
    result = snapshot.detection
    detection_age = now - snapshot.detection_at
    fresh_detection = (result is not None and detection_age <= 1.0
                       and snapshot.camera_status == "LIVE"
                       and (snapshot.detection_shape is None or snapshot.detection_shape == frame.shape[:2]))
    output = annotate(frame, result if fresh_detection else DetectionResult((), 0))
    camera_status = snapshot.camera_status
    if snapshot.frame is not None and now - snapshot.captured_at > 1.0 and camera_status == "LIVE":
        camera_status = "STALE"
    lines = [f"STATE: {snapshot.state.value} | CAMERA: {camera_status}"]
    telemetry = snapshot.telemetry
    if telemetry is None:
        lines.append("AGL / GPS: NO TELEMETRY")
    else:
        age = now - snapshot.telemetry_at
        suffix = " | STALE" if age > 1.0 else ""
        lines.extend((f"AGL: {telemetry.altitude_agl_m:.2f} m{suffix}",
                      f"GPS: {telemetry.position.latitude:.6f}, {telemetry.position.longitude:.6f}"))
    if result is None:
        lines.append("DETECTION: WAITING FOR ANALYSIS")
    elif not fresh_detection:
        lines.append(f"DETECTION: STALE ({detection_age:.1f}s)")
    else:
        lines.append(f"CIRCLES: {len(result.targets)} | QUADS: {result.quadrilateral_count} | age: {detection_age:.2f}s")
        lines.append(f"ANALYSIS: {snapshot.detection_mode}")
    # Darken only the small text area, keeping the underlying scene visible.
    panel_width = min(output.shape[1], 20 + max(
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines))
    panel_height = min(output.shape[0], 10 + 24 * len(lines))
    panel = output[:panel_height, :panel_width]
    panel[:] = cv2.convertScaleAbs(panel, alpha=0.25)
    for index, line in enumerate(lines):
        position = (10, 22 + 24 * index)
        cv2.putText(output, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return output


class LiveVideo:
    """Explicitly started HTTP server and latest-frame store.

    Optional get_telemetry must return a thread-safe cached sample promptly; it
    must not compete with execute() for a serial connection. No call is made by
    the camera or HTTP threads. A real adapter supplies that cache itself.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, *, fps: float = 30,
                 get_telemetry: Callable[[], UAVTelemetry] | None = None) -> None:
        if not math.isfinite(fps) or not 0 < fps <= 120:
            raise ValueError("Stream FPS must be between 0 and 120")
        if not 0 <= port <= 65535:
            raise ValueError("Invalid stream port")
        self.host, self.port, self.fps = host, port, fps
        self._get_telemetry = get_telemetry
        self._condition = threading.Condition(threading.Lock())
        self._snapshot = VideoSnapshot(None, 0, 0, MissionState.INIT, None, 0, None, 0, "WAITING")
        self._jpeg: bytes | None = None
        self._jpeg_sequence = 0
        self._encoding_failed = False
        self._stop = threading.Event()
        self._server: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []
        self._started = False

    def snapshot(self) -> VideoSnapshot:
        with self._condition:
            return self._snapshot

    def _update(self, **changes: object) -> None:
        with self._condition:
            self._snapshot = replace(self._snapshot, **changes)
            self._condition.notify_all()

    def set_state(self, state: MissionState) -> None:
        self._update(state=state)

    def set_camera_status(self, status: str) -> None:
        self._update(camera_status=status)

    def set_telemetry(self, telemetry: UAVTelemetry | None) -> None:
        self._update(telemetry=telemetry, telemetry_at=time.monotonic())

    def set_detection(self, result: DetectionResult, captured_at: float, *, sequence: int = 0,
                      shape: tuple[int, int] | None = None, mode: str = "DETECTED") -> None:
        self._update(detection=result, detection_at=captured_at, detection_sequence=sequence,
                     detection_shape=shape, detection_mode=mode)

    def wait_for_detection(self, sequence: int, timeout_s: float) -> VideoSnapshot:
        """Mission consumers require a result from their frame or a newer frame."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._snapshot.detection_sequence >= sequence or self._stop.is_set()
                or self._snapshot.camera_status in ("ERROR", "STOPPED"), timeout_s)
            if not ready:
                raise TimeoutError("Timed out waiting for analysis")
            return self._snapshot

    def publish_frame(self, frame: np.ndarray) -> None:
        """Own one immutable copy; producers never queue frames behind consumers."""
        if (not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8
                or not (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] in (3, 4)))):
            raise ValueError("Stream requires a non-empty uint8 grayscale/BGR/BGRA frame")
        owned = frame[:, :, :3].copy() if frame.ndim == 3 else frame.copy()
        owned.setflags(write=False)
        with self._condition:
            self._snapshot = replace(self._snapshot, frame=owned, sequence=self._snapshot.sequence + 1,
                                     captured_at=time.monotonic(), camera_status="LIVE")
            self._condition.notify_all()

    def wait_for_frame(self, after: int, timeout_s: float) -> VideoSnapshot:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._snapshot.sequence > after or self._snapshot.camera_status in ("ERROR", "STOPPED")
                or self._stop.is_set(), timeout=timeout_s)
            if self._snapshot.camera_status in ("ERROR", "STOPPED") or self._stop.is_set():
                raise RuntimeError(f"Camera stream unavailable: {self._snapshot.camera_status}")
            if not ready:
                raise TimeoutError("Timed out waiting for a camera frame")
            return self._snapshot

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/video_feed"

    def start(self) -> None:
        """Bind before starting workers; import/construction never opens sockets."""
        if self._started:
            raise RuntimeError("Create a new LiveVideo for each run")
        self._started = True
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body = (b"<!doctype html><title>UAV live video</title>"
                            b'<body style="margin:0;background:#111;color:white">'
                            b'<img src="/video_feed" alt="UAV video" style="max-width:100%;height:auto">'
                            b"</body>")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/video_feed":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.end_headers()
                sequence = 0
                try:
                    while not monitor._stop.is_set():
                        with monitor._condition:
                            monitor._condition.wait_for(
                                lambda: monitor._jpeg_sequence > sequence or monitor._stop.is_set()
                                or monitor._encoding_failed, timeout=1)
                            if monitor._stop.is_set() or monitor._encoding_failed:
                                break
                            jpeg, current = monitor._jpeg, monitor._jpeg_sequence
                        if jpeg is None or current == sequence:
                            continue
                        sequence = current
                        # Never hold a shared lock during socket writes.
                        header = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n"
                                  f"X-Sequence: {sequence}\r\n\r\n").encode("ascii")
                        self.wfile.write(header + jpeg + b"\r\n")
                except (OSError, TimeoutError):
                    pass  # A closed or slow browser must not affect the mission.

            def setup(self) -> None:
                self.request.settimeout(1.0)
                super().setup()

            def log_message(self, format: str, *args: object) -> None:
                logger.debug("HTTP %s", format % args)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server = server
        server.daemon_threads = True
        self.port = server.server_port
        workers: list[tuple[str, Callable[[], None]]] = [
            ("mjpeg-http", lambda: server.serve_forever(poll_interval=0.1)),
            ("mjpeg-encode", self._encode_loop),
        ]
        if self._get_telemetry is not None:
            workers.append(("video-telemetry", self._telemetry_loop))
        for name, target in workers:
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()
        logger.info("Live video: %s (up to %g FPS)", self.url, self.fps)

    def _encode_loop(self) -> None:
        next_frame_at = time.monotonic()
        last_warning = -math.inf
        while not self._stop.is_set():
            try:
                output = draw_overlay(self.snapshot())
                ok, data = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                jpeg = data.tobytes()
                with self._condition:
                    self._jpeg, self._jpeg_sequence = jpeg, self._jpeg_sequence + 1
                    self._condition.notify_all()
            except (cv2.error, ValueError):
                if time.monotonic() - last_warning >= 5:
                    logger.warning("Skipping invalid overlay/JPEG frame", exc_info=True)
                    last_warning = time.monotonic()
            except Exception:
                logger.exception("Video encoding stopped")
                with self._condition:
                    self._encoding_failed = True
                    self._condition.notify_all()
                break
            next_frame_at = max(next_frame_at + 1 / self.fps, time.monotonic())
            self._stop.wait(max(0, next_frame_at - time.monotonic()))

    def _telemetry_loop(self) -> None:
        assert self._get_telemetry is not None
        while not self._stop.is_set():
            try:
                self.set_telemetry(self._get_telemetry())
            except Exception:
                self.set_telemetry(None)
                logger.debug("Video telemetry unavailable", exc_info=True)
            self._stop.wait(0.1)

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        for thread in self._threads:
            thread.join(timeout=2)
            if thread.is_alive():
                logger.error("Video worker did not stop: %s", thread.name)

    def __enter__(self) -> LiveVideo:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
