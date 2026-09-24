"""HSV profily značek včetně kruhového odstínu 0..179."""

import json
from pathlib import Path
import cv2
import numpy as np


def default_profiles():
    return {'green': [[[35, 70, 60], [90, 255, 255]]],
            'red': [[[0, 100, 80], [10, 255, 255]], [[170, 100, 80], [179, 255, 255]]]}


def sample_color(frame, x, y):
    """Vzorek 11×11, percentily a rezerva pro drobné změny osvětlení."""
    height, width = frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError('Klikněte dovnitř obrazu.')
    patch = frame[max(0, y-5):min(height, y+6), max(0, x-5):min(width, x+6)]
    pixels = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    if np.median(pixels[:, 1]) < 25 or np.median(pixels[:, 2]) < 25:
        raise ValueError('Vzorek je příliš šedý nebo tmavý. Vyberte osvětlený střed značky.')
    angles = pixels[:, 0] * (2 * np.pi / 180)
    vector = np.exp(1j * angles).mean()
    if abs(vector) < 0.9:
        raise ValueError('Vzorek obsahuje příliš rozdílné barvy. Klikněte uvnitř značky.')
    center = (np.angle(vector) * 180 / (2 * np.pi)) % 180
    offsets = (pixels[:, 0] - center + 90) % 180 - 90
    low_h = int(np.floor(center + np.percentile(offsets, 5) - 8))
    high_h = int(np.ceil(center + np.percentile(offsets, 95) + 8))
    if high_h - low_h > 50:
        raise ValueError('Vzorek je barevně nejednotný. Zvolte jiný bod.')
    low_sv = np.maximum([15, 15], np.floor(np.percentile(pixels[:, 1:], 5, axis=0) - 40)).astype(int).tolist()
    high_sv = np.minimum(255, np.ceil(np.percentile(pixels[:, 1:], 95, axis=0) + 40)).astype(int).tolist()
    if low_h < 0:
        hue_ranges = [(0, high_h), (180 + low_h, 179)]
    elif high_h > 179:
        hue_ranges = [(0, high_h - 180), (low_h, 179)]
    else:
        hue_ranges = [(low_h, high_h)]
    return [[[low, *low_sv], [high, *high_sv]] for low, high in hue_ranges]


def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, tuple(low), tuple(high))
    return mask


def profiles_overlap(profiles):
    return any(all(max(a, c) <= min(b, d) for a, b, c, d in zip(low1, high1, low2, high2))
               for low1, high1 in profiles['green'] for low2, high2 in profiles['red'])


def validate_profiles(profiles):
    if not isinstance(profiles, dict) or set(profiles) != {'green', 'red'}:
        raise ValueError('Kalibrace musí obsahovat green a red.')
    for ranges in profiles.values():
        if not isinstance(ranges, list) or not 1 <= len(ranges) <= 2:
            raise ValueError('Neplatný počet HSV rozsahů.')
        for pair in ranges:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Neplatný HSV rozsah.')
            low, high = pair
            if not all(isinstance(row, list) and len(row) == 3 for row in (low, high)):
                raise ValueError('HSV mez musí mít tři hodnoty.')
            for a, b, limit in zip(low, high, (179, 255, 255)):
                if type(a) is not int or type(b) is not int or not 0 <= a <= b <= limit:
                    raise ValueError('HSV hodnoty jsou mimo povolený rozsah.')
    return profiles


def load_profiles(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('Neznámá verze kalibrace.')
    return validate_profiles(data.get('profiles'))


def save_profiles(path, profiles):
    validate_profiles(profiles)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps({'version': 1, 'profiles': profiles}, indent=2), encoding='utf-8')
    temporary.replace(path)
