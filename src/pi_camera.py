"""CSI kamera Raspberry Pi přes systémový balíček Picamera2."""

import time
import math


class PiCamera:
    def __init__(self, index=0, *, ev=None, width=640, height=480, tuning_file=None, saturation=None):
        if width <= 0 or height <= 0:
            raise ValueError('Rozlišení musí být kladné.')
        self.size = (width, height)
        if ev is not None and (not math.isfinite(ev) or not -8 <= ev <= 8):
            raise ValueError('Expozice EV musí být v rozsahu -8 až 8.')
        self.index = index
        self.ev = ev
        self.tuning_file = tuning_file
        # Sytost barev v ISP (libcamera Saturation, výchozí 1.0): vyšší = červenější červená.
        if saturation is not None and (not math.isfinite(saturation) or not 0 <= saturation <= 32):
            raise ValueError('Sytost musí být v rozsahu 0 až 32.')
        self.saturation = saturation
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
        # Profil musí dostat úplně první volání libcamery (jako red_tracker.py).
        # Dřívější global_camera_info() spustí libcameru se systémovým ov5647.json
        # a pozdější tuning= se tiše ignoruje -> růžový obraz NoIR kamery.
        tuning = Picamera2.load_tuning_file(self.tuning_file) if self.tuning_file else None
        try:
            camera = Picamera2(camera_num=self.index, tuning=tuning)
        except IndexError as error:
            raise RuntimeError(f'CSI kamera {self.index} není dostupná. Ověřte rpicam-hello --list-cameras.') from error
        try:
            # RGB888 v Picamera2 poskytuje bajty B,G,R, jak je očekává OpenCV.
            config = camera.create_video_configuration(
                main={'size': self.size, 'format': 'RGB888'}, buffer_count=3)
            camera.configure(config)
            # Stejně jako red_tracker.py ponecháme výchozí řízení z profilu.
            # Expozici měníme pouze při explicitním --ev.
            controls = {}
            if self.ev is not None:
                controls.update({'AeEnable': True, 'ExposureValue': self.ev})
            if self.saturation is not None:
                controls['Saturation'] = self.saturation
            if controls:
                camera.set_controls(controls)
            camera.start()
            time.sleep(2)  # Ustálení automatické expozice před prvním snímkem.
            self._report(camera)
        except BaseException:
            camera.close()
            raise
        self._camera = camera

    def _report(self, camera):
        # Pro porovnání s red_tracker.py: bez profilu NoIR jsou zisky R/B jiné a obraz růžový.
        import sys
        try:
            metadata = camera.capture_metadata()
            gains = tuple(round(float(g), 2) for g in metadata['ColourGains'])
            colour = f'ColourGains {gains}, ColourTemperature {metadata.get("ColourTemperature", "?")} K'
        except Exception:
            colour = 'vyvážení barev neznámé'
        print(f'Kamera: požadovaný profil {self.tuning_file or "systémový (bez NoIR)"} '
              f'(skutečný ukazuje řádek libcamery "Using tuning file"), '
              f'{self.size[0]}x{self.size[1]}, {colour}', file=sys.stderr)

    def set_saturation(self, value):
        """Změna sytosti za běhu (posuvník v --tune)."""
        self.saturation = value
        if self._camera is not None:
            self._camera.set_controls({'Saturation': value})

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
