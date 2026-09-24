"""Klávesy a kliknutí pro učení barev z původního snímku bez popisků."""

import cv2
from .color_calibration import sample_color, save_profiles, default_profiles, profiles_overlap


class CalibrationUI:
    def __init__(self, vision, target, path):
        self.vision, self.target, self.path = vision, target, path
        self.frame = None
        self.mode = None
        self.dirty = False
        self.message = 'G: zelena | C: cervena | S: ulozit | D: vychozi | M: rucni bod'

    def on_mouse(self, event, x, y, flags, userdata):
        if self.mode is None:
            self.target.on_mouse(event, x, y, flags, userdata)
        elif event == cv2.EVENT_LBUTTONDOWN and self.frame is not None:
            try:
                ranges = sample_color(self.frame, x, y)
                self.vision.detector.profiles[self.mode] = ranges
                self.vision.reset()
                self.dirty = True
                self.message = f'{self.mode}: vzorek nastaven. S = ulozit; M = rucni bod'
                print(f'Kalibrace {self.mode}: {ranges}', flush=True)
            except ValueError as error:
                self.message = 'Vzorek odmitnut. Vyberte osvetleny stred znacky.'
                print(f'Kalibrace: {error}', flush=True)

    def handle_key(self, key):
        if key in (ord('g'), ord('G'), ord('c'), ord('C')):
            self.mode = 'green' if key in (ord('g'), ord('G')) else 'red'
            self.message = f'{self.mode}: kliknete na stred fyzicke znacky (vzorek 11x11).'
        elif key in (ord('m'), ord('M')):
            self.mode = None
            self.message = 'Rucni bod. G: zelena | C: cervena | S: ulozit | D: vychozi'
        elif key in (ord('s'), ord('S')):
            try:
                save_profiles(self.path, self.vision.detector.profiles)
                self.dirty = False
                self.message = 'Kalibrace ulozena.'
                print(f'Kalibrace uložena: {self.path}', flush=True)
            except OSError as error:
                self.message = 'Ulozeni selhalo; podrobnosti v terminalu.'
                print(f'Kalibraci nelze uložit: {error}', flush=True)
        elif key in (ord('d'), ord('D')):
            self.vision.detector.profiles = default_profiles()
            self.vision.reset()
            self.dirty = True
            self.message = 'Vychozi barvy obnoveny. S = ulozit.'

    def annotate(self, image):
        cv2.putText(image, self.message, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 255, 255), 1)
        if self.dirty:
            cv2.putText(image, 'NEULOZENO (S)', (10, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1)
        if profiles_overlap(self.vision.detector.profiles):
            cv2.putText(image, 'POZOR: barvy se prekryvaji - zkalibrujte obe znacky',
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1)
        return image
