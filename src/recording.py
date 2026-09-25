"""Volitelný úplný záznam zpracovaných snímků s původními časy příjmu."""
import json
from pathlib import Path
import tempfile
import cv2


class FrameRecorder:
    def __init__(self, root):
        Path(root).mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='sequence-', dir=root))
        self.stream = None
        self.first_time = None
        self.count = 0

    def __enter__(self):
        self.stream = (self.directory/'manifest.jsonl').open('w', encoding='utf-8')
        return self

    def write(self, frame, sample_time):
        if self.first_time is None:
            self.first_time = sample_time
        name = f'{self.count:06d}.png'
        ok, encoded = cv2.imencode('.png', frame)
        if not ok:
            raise RuntimeError('Nelze uložit záznam snímku.')
        encoded.tofile(self.directory/name)
        # None není negativní snímek. Anotátor musí označit cíle nebo explicitně [].
        item = dict(image=name, time_s=sample_time-self.first_time, targets=None)
        self.stream.write(json.dumps(item, allow_nan=False)+'\n')
        self.stream.flush()
        self.count += 1

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.close()
