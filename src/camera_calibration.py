"""Kalibrace kamery šachovnicí: ohnisková vzdálenost v pixelech a zkreslení objektivu."""
import cv2
import numpy as np

from .geometry import CameraModel

MIN_VIEWS = 10


def find_corners(frame, pattern):
    """Vnitřní rohy šachovnice (sloupce, řádky) se subpixelovou přesností, nebo None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners if found else None


def calibrate(corner_sets, pattern, square, size):
    """-> (CameraModel, RMS chyba reprojekce v px). square je strana pole v metrech."""
    if len(corner_sets) < MIN_VIEWS:
        raise ValueError(f'Potřeba alespoň {MIN_VIEWS} snímků šachovnice, je {len(corner_sets)}.')
    board = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        [board] * len(corner_sets), corner_sets, tuple(size), None, None)
    return CameraModel(matrix, distortion.ravel(), tuple(size)), rms
