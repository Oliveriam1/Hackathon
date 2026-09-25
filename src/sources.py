"""Otevření obrazového zdroje a fotografie; prostředky vlastní ExitStack aplikace."""
import cv2
import numpy as np
from .camera import Camera
from .pi_camera import PiCamera


def open_source(config, stack):
    if config.demo:
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (220, 220), (0, 200, 0), -1)
        cv2.circle(frame, (460, 300), 45, (0, 0, 230), -1)
        cv2.rectangle(frame, (370, 210), (570, 400), (220, 220, 220), 3)
        return None, frame, 'static'
    if config.image is not None:
        frame = cv2.imdecode(np.fromfile(config.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError('Soubor nelze načíst jako obrázek.')
        return None, frame, 'static'
    if config.video is not None:
        if not config.video.is_file():
            raise RuntimeError('Videosoubor neexistuje.')
        device = Camera(str(config.video))
    elif config.picamera is not None:
        device = PiCamera(config.picamera, ev=config.ev, saturation=config.saturation,
                          width=config.width, height=config.height,
                          tuning_file=None if config.tuning_file == 'none' else
                          config.tuning_file or 'ov5647_noir.json')
    else:
        device = Camera(config.camera)
    camera = stack.enter_context(device)
    return camera, camera.read(), 'video' if config.video is not None else 'camera'


def save_snapshot(frame, path):
    extension = path.suffix.lower()
    if extension not in ('.jpg', '.jpeg', '.png'):
        raise RuntimeError('Snímek musí mít příponu .jpg, .jpeg nebo .png.')
    success, encoded = cv2.imencode(extension, frame)
    if not success:
        raise RuntimeError('Snímek se nepodařilo zakódovat.')
    encoded.tofile(path)
    print(f'Snímek uložen: {path.resolve()}')
