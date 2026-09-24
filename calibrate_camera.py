"""Kalibrace kamery šachovnicí pro výpočet polohy kolečka.

Vytiskněte šachovnici (výchozí 10 x 7 polí = 9 x 6 vnitřních rohů), nalepte ji
na rovnou desku a ukazujte ji kameře z různých úhlů a vzdáleností, i v rozích obrazu.
Snímek se uloží sám, když je šachovnice vidět a posunula se. Kalibrace platí jen
pro stejné rozlišení a režim kamery, jaký používá detektor (640 x 480).
"""
import argparse
import glob
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.camera_calibration import MIN_VIEWS, calibrate, find_corners
from src.pi_camera import PiCamera


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--picamera', type=int, metavar='INDEX', help='CSI kamera přes Picamera2.')
    source.add_argument('--camera', type=int, metavar='INDEX', help='Kamera přes OpenCV.')
    source.add_argument('--images', help='Uložené snímky, např. "kalibrace/*.png".')
    parser.add_argument('--pattern', default='9x6', help='Vnitřní rohy šachovnice SLOUPCExŘÁDKY (výchozí 9x6).')
    parser.add_argument('--square', type=float, default=25.0, help='Strana pole v mm (výchozí 25).')
    parser.add_argument('--count', type=int, default=25, help='Počet snímků z kamery (výchozí 25).')
    parser.add_argument('--save-dir', type=Path, help='Uloží použité snímky pro pozdější --images.')
    parser.add_argument('--output', type=Path, default=Path('camera_calibration.json'))
    args = parser.parse_args()
    try:
        pattern = tuple(int(value) for value in args.pattern.lower().split('x'))
        if len(pattern) != 2 or min(pattern) < 3:
            raise ValueError
    except ValueError:
        parser.error('--pattern musí být např. 9x6.')
    if args.count < MIN_VIEWS:
        parser.error(f'--count musí být alespoň {MIN_VIEWS}.')

    try:
        frames = load_images(args.images) if args.images else capture(args, pattern)
        corner_sets = [corners for corners in (find_corners(frame, pattern) for frame in frames) if corners is not None]
        sizes = {frame.shape[1::-1] for frame in frames}
        if len(sizes) != 1:
            raise RuntimeError('Všechny snímky musí mít stejné rozlišení.')
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(frames):
                cv2.imencode('.png', frame)[1].tofile(args.save_dir / f'sachovnice_{index:02d}.png')
        camera, rms = calibrate(corner_sets, pattern, args.square / 1000, sizes.pop())
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f'Chyba kalibrace: {error}', file=sys.stderr)
        return 1
    camera.save(args.output, rms_px=round(rms, 3), views=len(corner_sets))
    print(f'Uloženo {args.output.resolve()}: {len(corner_sets)} snímků, RMS {rms:.2f} px, '
          f'fx {camera.matrix[0, 0]:.1f} px, zorné pole {camera.fov[0]:.1f}° x {camera.fov[1]:.1f}°')
    if rms > 1.0:
        print('Varování: RMS nad 1 px, zopakujte kalibraci s ostřejšími snímky.', file=sys.stderr)
    return 0


def load_images(pattern):
    paths = sorted(glob.glob(pattern))
    frames = [cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError('Snímky nelze načíst.')
    return frames


def capture(args, pattern):
    show = not sys.platform.startswith('linux') or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    frames, last_time, last_center = [], 0.0, None
    with ExitStack() as stack:
        device = PiCamera(args.picamera) if args.picamera is not None else Camera(args.camera)
        camera = stack.enter_context(device)
        if show:
            stack.callback(cv2.destroyAllWindows)
        while len(frames) < args.count:
            frame = camera.read()
            corners = find_corners(frame, pattern)
            if corners is not None:
                center = corners[:, 0, :].mean(axis=0)
                # Nový snímek jen po posunu šachovnice: rozmanité polohy zpřesní kalibraci.
                moved = last_center is None or np.linalg.norm(center - last_center) > 40
                if moved and time.monotonic() - last_time > 1.0:
                    frames.append(frame.copy())
                    last_time, last_center = time.monotonic(), center
                    print(f'Snímek {len(frames)}/{args.count}', flush=True)
            if show:
                preview = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(preview, pattern, corners, True)
                cv2.putText(preview, f'{len(frames)}/{args.count}  Q: konec', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Kalibrace', preview)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    return frames


if __name__ == '__main__':
    raise SystemExit(main())
