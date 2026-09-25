"""Ladění bez okna: každé 2 s uloží surový a označený snímek + stav (nejvýše 30 sad)."""
import json
from pathlib import Path
import tempfile
import time
import cv2
from .vision import annotate


class Diagnostics:
    def __init__(self, directory, *, every_s=2., limit=30, clock=time.monotonic):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='run-', dir=directory))
        self.every_s, self.limit, self.clock = every_s, limit, clock
        self.saved = 0
        self.last_saved = None

    def update(self, frame, observation, aim, record):
        now = self.clock()
        if self.saved >= self.limit or (self.last_saved is not None and now-self.last_saved < self.every_s):
            return False
        prefix = self.directory / f'{self.saved:03d}'
        for suffix, image in (('raw', frame), ('marked', annotate(frame, observation, aim))):
            success, encoded = cv2.imencode('.png', image)
            if not success:
                raise RuntimeError('Nelze uložit diagnostický snímek.')
            encoded.tofile(f'{prefix}-{suffix}.png')
        Path(f'{prefix}.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        self.saved += 1
        self.last_saved = now
        return True
