"""Okno náhledu pro ladění: kandidáti, cíl, stav míření, volitelně posuvníky prahů."""
import sys
import cv2
from .vision import annotate


class Tuner:
    """Posuvníky prahů červené a sytosti kamery; změny vypíše jako parametry příkazu."""

    def __init__(self, window, detector, camera):
        self.window, self.detector = window, detector
        self.camera = camera if hasattr(camera, 'set_saturation') else None
        cv2.createTrackbar('REDNESS_MIN', window, detector.redness_min, 255, lambda _: None)
        cv2.createTrackbar('R_MIN', window, detector.r_min, 255, lambda _: None)
        cv2.createTrackbar('RED_FRACTION x100', window, round(detector.red_fraction_min*100), 100, lambda _: None)
        if self.camera is not None:
            cv2.createTrackbar('SATURATION x10', window, round((self.camera.saturation or 1.0)*10), 60, lambda _: None)
        self.last = None

    def update(self):
        values = (cv2.getTrackbarPos('REDNESS_MIN', self.window), cv2.getTrackbarPos('R_MIN', self.window),
                  cv2.getTrackbarPos('RED_FRACTION x100', self.window)/100,
                  cv2.getTrackbarPos('SATURATION x10', self.window)/10 if self.camera is not None else None)
        if values == self.last:
            return False
        self.detector.redness_min, self.detector.r_min, self.detector.red_fraction_min = values[:3]
        if self.camera is not None and (self.last is None or values[3] != self.last[3]):
            self.camera.set_saturation(values[3])
        self.last = values
        command = f'--redness-min {values[0]} --r-min {values[1]} --red-fraction {values[2]:.2f}'
        if values[3] is not None:
            command += f' --saturation {values[3]:.1f}'
        print(f'Ladění: {command}', file=sys.stderr)
        return True


class Preview:
    def __init__(self, config, vision, camera):
        self.config, self.vision, self.camera = config, vision, camera
        self.window = 'Tecka - Q / Esc: konec, E: maska'
        self.show_mask = False
        self.tuner = None

    def __enter__(self):
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            if self.config.tune:
                self.tuner = Tuner(self.window, self.vision.detector, self.camera)
        except BaseException:
            cv2.destroyAllWindows()
            raise
        return self

    def update(self, frame, observation, aim=None):
        """Vykreslí snímek; vrací False, když uživatel okno zavřel."""
        mask = self.vision.detector.edges
        background = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if self.show_mask and mask is not None else frame
        cv2.imshow(self.window, annotate(background, observation, aim))
        key = cv2.waitKey(1) & 0xFF
        if self.tuner is not None and self.tuner.update():
            self.vision.reset()
        if key in (ord('e'), ord('E')):
            self.show_mask = not self.show_mask
        if key in (ord('q'), ord('Q'), 27):
            return False
        return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1

    def __exit__(self, *args):
        cv2.destroyAllWindows()
