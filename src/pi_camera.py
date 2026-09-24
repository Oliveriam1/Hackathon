"""CSI kamera Raspberry Pi přes systémový balíček Picamera2."""

import time
import math


class PiCamera:
    def __init__(self, index=0, *, ev=0.0):
        if not math.isfinite(ev) or not -8 <= ev <= 8:
            raise ValueError('Expozice EV musí být v rozsahu -8 až 8.')
        self.index = index
        self.ev = ev
        self._camera = None

    def open(self):
        if self._camera is not None:
            return
        try:
            from picamera2 import Picamera2
        except ImportError as error:
            raise RuntimeError(
                'Picamera2 není dostupné. Na Pi nainstalujte python3-picamera2 '
                'a použijte venv s --system-site-packages.'
            ) from error
        cameras = Picamera2.global_camera_info()
        if not 0 <= self.index < len(cameras):
            raise RuntimeError(f'CSI kamera {self.index} není dostupná. Ověřte rpicam-hello --list-cameras.')
        camera = Picamera2(camera_num=self.index)
        try:
            # RGB888 v Picamera2 poskytuje bajty B,G,R, jak je očekává OpenCV.
            config = camera.create_preview_configuration(
                main={'size': (640, 480), 'format': 'RGB888'})
            camera.configure(config)
            # Libcamera AwbMode Auto = 0; EV upravuje cíl automatické expozice.
            camera.set_controls({'AeEnable': True, 'ExposureValue': self.ev,
                                 'AwbEnable': True, 'AwbMode': 0})
            camera.start()
            time.sleep(2)  # Ustálení automatické expozice před prvním snímkem.
        except BaseException:
            camera.close()
            raise
        self._camera = camera

    def read(self):
        if self._camera is None:
            raise RuntimeError('CSI kamera není otevřená.')
        frame = self._camera.capture_array('main')
        if frame is None or frame.size == 0:
            raise RuntimeError('CSI kamera nevrátila snímek.')
        return frame

    def close(self):
        if self._camera is not None:
            camera, self._camera = self._camera, None
            try:
                camera.stop()
            finally:
                camera.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
