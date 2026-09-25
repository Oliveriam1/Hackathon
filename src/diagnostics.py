"""Čitelný stav a omezený záznam důkazů pro ladění bez živého okna."""
import json
from pathlib import Path
import tempfile
import time
import cv2
from .vision import annotate_observation


class Diagnostics:
    def __init__(self, stream=None, directory=None, clock=time.monotonic):
        self.stream, self.clock = stream, clock
        self.last_report = None
        self.last_saved = None
        self.count = 0
        self.saved = 0
        self.directory = None
        if directory is not None:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix='run-', dir=directory))

    def update(self, frame, observation, record):
        now = self.clock()
        self.count += 1
        if self.stream is not None and (self.last_report is None or now-self.last_report >= 1):
            fps = '--' if self.last_report is None else f'{self.count/(now-self.last_report):.1f}'
            lock = record['visual_lock']['state']
            kind = 'MERENI' if observation.measured else ('PREDIKCE' if observation.target else 'ZADNY')
            point = observation.measurement
            position = f'({point.x:.0f}, {point.y:.0f})' if point is not None else '--'
            stage = ('BEZ_4UHELNIKU' if not observation.rectangles else
                     'BEZ_KOLECKA_UVNITR' if not observation.circles else
                     'VICE_KANDIDATU' if len(observation.circles) > 1 else 'KANDIDAT')
            timings = observation.stage_ms or {}
            detail = '/'.join(f'{timings.get(key, 0):.0f}' for key in
                              ('prepare', 'quadrilaterals', 'circles_and_tracking'))
            if observation.detector_mode == 'red':
                debug = observation.red or {}
                stage = debug.get('tracking', {}).get('state', 'SEARCHING')
                rejected = debug.get('diagnostics', {}).get('rejections', {})
                stage += ' odmitnuto=' + (','.join(f'{k}:{v}' for k, v in rejected.items()) or '--')
                detail = '/'.join(f'{timings.get(k, 0):.0f}' for k in
                                  ('mask', 'components', 'validation', 'geometry', 'tracking'))
            gimbal = record.get('gimbal', {})
            servo_text = ''
            if gimbal.get('enabled'):
                angles = gimbal['commanded_angles_deg']
                servo_text = f" | kamera={gimbal['state']} {angles['right']:+.1f}/{angles['forward']:+.1f} deg"
                if gimbal.get('dry_run'):
                    servo_text += ' DRY_RUN'
            print(f'FPS={fps} | detekce={observation.processing_ms:.0f} ms | '
                  f'4uhelniky={len(observation.rectangles)} | kandidati={len(observation.circles)} | '
                  f'{kind} {position} | {lock} | {stage} | faze_ms={detail}{servo_text}', file=self.stream, flush=True)
            self.last_report, self.count = now, 0
        if self.directory is not None and self.saved < 30 and (self.last_saved is None or now-self.last_saved >= 2):
            prefix = self.directory / f'{self.saved:03d}'
            for suffix, image in (('raw', frame), ('marked', annotate_observation(frame, observation))):
                success, encoded = cv2.imencode('.png', image)
                if not success:
                    raise RuntimeError('Nelze uložit diagnostický snímek.')
                encoded.tofile(str(prefix) + f'-{suffix}.png')
            Path(str(prefix) + '.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
            self.saved += 1
            self.last_saved = now
