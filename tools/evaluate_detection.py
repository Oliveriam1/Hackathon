"""Offline vyhodnocení anotovaných sekvencí, bez kamery a bez povelů.

python tools/evaluate_detection.py manifest.json --output report.json
Manifest: {"frames": [{"image": "000.png", "time_s": 0,
"sequence": "pole", "targets": [{"id": "terc", "x": 100, "y": 80, "radius": 6}]}]}
Prázdné targets znamená negativní snímek. Čas musí být v sekvenci rostoucí.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision import Vision


def evaluate(manifest_path, expected_diameter=None):
    manifest_path = Path(manifest_path)
    text = manifest_path.read_text(encoding='utf-8')
    data = {'frames': [json.loads(line) for line in text.splitlines() if line.strip()]} if manifest_path.suffix == '.jsonl' else json.loads(text)
    tp = fp = fn = false_locks = switches = 0
    times, errors, details = [], [], []
    previous_sequence = object()
    previous_time = None
    identities = {}
    for item in data['frames']:
        sequence = item.get('sequence', 'default')
        if sequence != previous_sequence:
            vision = Vision('red', expected_diameter=expected_diameter)
            previous_sequence, previous_time, identities = sequence, None, {}
        timestamp = float(item['time_s'])
        if not math.isfinite(timestamp) or (previous_time is not None and timestamp <= previous_time):
            raise ValueError('Časy musí v každé sekvenci růst.')
        previous_time = timestamp
        path = manifest_path.parent / item['image']
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'Nelze načíst {path}')
        observation = vision.observe(frame, sample_time=timestamp)
        targets = item['targets']
        if targets is None:
            raise ValueError(f"Snímek {item['image']} není anotovaný: doplňte targets, nebo [] bez cíle.")
        times.append(observation.processing_ms)
        # Jednoznačné přiřazení kandidátů k anotacím, nejbližší dvojice nejdřív.
        pairs = sorted((math.hypot(c.x-t['x'], c.y-t['y']), ci, ti)
                       for ci, c in enumerate(observation.circles) for ti, t in enumerate(targets))
        used_c, used_t = set(), set()
        for distance, ci, ti in pairs:
            if ci not in used_c and ti not in used_t and distance <= max(3., targets[ti]['radius']):
                used_c.add(ci)
                used_t.add(ti)
                errors.append(distance)
        tp += len(used_c)
        fp += len(observation.circles)-len(used_c)
        fn += len(targets)-len(used_t)
        if observation.confirmed and observation.measured:
            c = observation.measurement
            matches = [(math.hypot(c.x-t['x'], c.y-t['y']), t) for t in targets]
            matches.sort(key=lambda x: x[0])
            if not matches or matches[0][0] > max(3., matches[0][1]['radius']):
                false_locks += 1
            else:
                identity = matches[0][1]['id']
                track_id = observation.red['tracking']['track_id']
                if track_id in identities and identities[track_id] != identity:
                    switches += 1
                identities[track_id] = identity
        details.append({'image': item['image'], 'state': observation.red['tracking']['state'],
                        'candidate_count': len(observation.circles), 'processing_ms': observation.processing_ms,
                        'stages_ms': observation.stage_ms})
    return dict(frames=len(times), true_positives=tp, false_positives=fp, missed_targets=fn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                false_confirmed_frames=false_locks, identity_switches=switches,
                center_error_px_mean=float(np.mean(errors)) if errors else None,
                processing_ms_median=float(np.median(times)) if times else None,
                processing_ms_p95=float(np.percentile(times, 95)) if times else None,
                results=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--red-diameter-px', type=float)
    args = parser.parse_args()
    if args.red_diameter_px is not None and (not math.isfinite(args.red_diameter_px) or args.red_diameter_px <= 0):
        parser.error('Průměr musí být kladný.')
    report = evaluate(args.manifest, args.red_diameter_px)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        if args.output.resolve() == args.manifest.resolve():
            parser.error('Výstup nesmí přepsat manifest.')
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
