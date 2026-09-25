"""Životní cyklus aplikace: zdroje, zpracování, výstup a ukončení."""
import sys
import os
import time
from contextlib import ExitStack
import cv2
from .config import AppConfig
from .sources import open_source, save_snapshot
from .preview import Preview
from .pipeline import DetectionPipeline
from .publisher import JSONPublisher
from .diagnostics import Diagnostics
from .recording import FrameRecorder


def run(args: AppConfig) -> int:
    try:
        if not args.snapshot and not args.headless and sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('Není dostupná grafická plocha. Přes SSH použijte --snapshot test.jpg.')
        with ExitStack() as stack:
            telemetry = None
            if args.mavlink:
                from src.telemetry import MAVLinkTelemetry
                telemetry = stack.enter_context(MAVLinkTelemetry(args.mavlink, args.baud, args.target_system))
            gimbal = None
            if args.servo:
                # red_tracker.py zapíná serva před kamerou.
                from src.gimbal import Gimbal
                try:
                    gimbal = stack.enter_context(Gimbal())
                except RuntimeError as error:
                    if not args.jako_red_tracker:
                        raise
                    print(f'Varování: pokračuji bez serv. {error}', file=sys.stderr)
            camera, frame, source_name = open_source(args, stack)
            received_at = time.time()
            sample_time = time.monotonic()
            if args.snapshot:
                save_snapshot(frame, args.snapshot)
                return 0
            pipeline = DetectionPipeline(args, telemetry=telemetry, gimbal=gimbal)
            recorder = stack.enter_context(FrameRecorder(args.record_dir)) if args.record_dir else None
            if recorder is not None:
                print(f'Záznam: {recorder.directory}', file=sys.stderr)
            vision = pipeline.vision
            stream = stack.enter_context(args.output.open('a', encoding='utf-8')) if args.output else sys.stdout
            publisher = JSONPublisher(stream) if args.output or not args.status else None
            diagnostics = Diagnostics(stream=sys.stdout if args.status else None, directory=args.diagnostics)
            if diagnostics.directory is not None:
                print(f'Diagnostika: {diagnostics.directory}', file=sys.stderr)
            preview = None if args.headless else stack.enter_context(Preview(args, vision, camera))
            observation = None
            while True:
                if observation is None or camera is not None:
                    observation, record = pipeline.process(frame, received_at=received_at,
                                                           sample_time=sample_time, source=source_name)
                    if publisher is not None:
                        publisher.publish(record['drone_data'] if args.drone_data else record)
                    diagnostics.update(frame, observation, record)
                    if recorder is not None:
                        recorder.write(frame, sample_time)
                if args.frames is not None and pipeline.sequence >= args.frames:
                    break
                if args.headless:
                    if camera is None:
                        break  # Statický obrázek není několik nezávislých měření.
                    try:
                        frame = camera.read()
                    except EOFError:
                        break
                    received_at = time.time()
                    sample_time = time.monotonic()
                    continue
                keep_running, reprocess = preview.update(frame, observation)
                if not keep_running:
                    break
                if camera is None and reprocess:
                    observation = None
                if camera is not None:
                    try:
                        frame = camera.read()
                        received_at = time.time()
                        sample_time = time.monotonic()
                    except EOFError:
                        break  # konec záznamu
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"Chyba náhledu: {error}", file=sys.stderr)
        return 1
    return 0

