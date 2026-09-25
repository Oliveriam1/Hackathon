"""Geometrie kamery: model objektivu, natočení závěsu a dronu, převody úhlů.

Souřadnice: sever/východ/dolů (NED). Tělo dronu: dopředu/doprava/dolů.
Závěs: vnější servo (osa x, "right") a vnitřní servo (osa y, "forward").
0/0 = kamera míří kolmo dolů. Kladné right = kamera kouká doprava,
kladné forward = kamera kouká dopředu.
"""
from dataclasses import dataclass
import json
import math

import cv2
import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
# OV5647 (2592 x 1944 px po 1.4 um) s objektivem 3.6 mm, režim s celým zorným polem.
NOMINAL_FOV = (53.5, 41.4)


@dataclass(frozen=True)
class DronePose:
    lat: float          # stupně WGS84
    lon: float
    height: float       # m nad zemí, na které leží tečka
    roll: float = 0.0   # stupně, kladně = pravé rameno dolů
    pitch: float = 0.0  # stupně, kladně = příď nahoru
    yaw: float = 0.0    # stupně od severu po směru hodinových ručiček


@dataclass(frozen=True)
class GimbalAngles:
    """Skutečné úhly kamery vůči tělu dronu, 0/0 = kolmo dolů."""
    right: float = 0.0
    forward: float = 0.0


@dataclass(frozen=True, eq=False)
class CameraModel:
    matrix: np.ndarray      # 3x3: fx, fy, cx, cy v pixelech
    distortion: np.ndarray  # k1, k2, p1, p2, k3 (OpenCV)
    size: tuple[int, int]   # (šířka, výška), pro kterou platí kalibrace

    @classmethod
    def from_fov(cls, size=(640, 480), fov=NOMINAL_FOV):
        """Jmenovitý model bez zkreslení; jen dokud není kalibrace."""
        width, height = size
        fx = width / 2 / math.tan(math.radians(fov[0]) / 2)
        fy = height / 2 / math.tan(math.radians(fov[1]) / 2)
        return cls(np.array([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]]), np.zeros(5), tuple(size))

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as file:
            data = json.load(file)
        return cls(np.array(data['matrix'], float), np.array(data['distortion'], float), tuple(data['size']))

    def save(self, path, **info):
        data = {'matrix': self.matrix.tolist(), 'distortion': self.distortion.ravel().tolist(),
                'size': list(self.size), 'fov_deg': [round(value, 2) for value in self.fov], **info}
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    @property
    def fov(self):
        width, height = self.size
        return (math.degrees(2*math.atan(width/2/self.matrix[0, 0])),
                math.degrees(2*math.atan(height/2/self.matrix[1, 1])))

    def ray(self, u, v):
        """Směr paprsku pixelu v souřadnicích kamery (x vpravo, y dolů, z dopředu)."""
        # Iterace do konvergence: výchozích 5 kroků u silného zkreslení nestačí.
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)
        point = np.array([[[u, v]]], float)
        if hasattr(cv2, 'undistortPointsIter'):  # OpenCV 4.x
            x, y = cv2.undistortPointsIter(point, self.matrix, self.distortion, None, None, criteria)[0, 0]
        else:  # OpenCV 5.x
            x, y = cv2.undistortPoints(point, self.matrix, self.distortion, criteria=criteria)[0, 0]
        return np.array([x, y, 1.0])


def _rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Kamera -> závěs při 0/0: obraz vpravo = doprava, obraz dolů = dozadu, osa = dolů.
_CAMERA_IN_GIMBAL = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
IMAGE_TOP = {'forward': 0, 'right': 90, 'backward': 180, 'left': 270}


def camera_in_body(gimbal, image_top='forward'):
    """Rotace kamera -> tělo dronu (FRD: dopředu, doprava, dolů)."""
    return (_rx(-math.radians(gimbal.right)) @ _ry(math.radians(gimbal.forward))
            @ _rz(math.radians(IMAGE_TOP[image_top])) @ _CAMERA_IN_GIMBAL)


def body_in_ned(pose):
    return _rz(math.radians(pose.yaw)) @ _ry(math.radians(pose.pitch)) @ _rx(math.radians(pose.roll))


def offset_to_latlon(lat, lon, north, east):
    """Posun v metrech -> GPS; poloměry křivosti elipsoidu WGS84 v místě dronu."""
    sin_lat = math.sin(math.radians(lat))
    denominator = 1 - WGS84_E2 * sin_lat**2
    meridian = WGS84_A * (1 - WGS84_E2) / denominator**1.5
    normal = WGS84_A / math.sqrt(denominator)
    return (lat + math.degrees(north / meridian),
            lon + math.degrees(east / (normal * math.cos(math.radians(lat)))))


def centering_angles(pixel, gimbal, camera, image_top='forward'):
    """Úhly závěsu, při kterých se bod z pixelu dostane do středu obrazu."""
    x, y, z = camera_in_body(gimbal, image_top) @ camera.ray(*pixel)
    length = math.sqrt(x*x + y*y + z*z)
    # Osa kamery v těle dronu je (sin f, sin r * cos f, cos r * cos f).
    return GimbalAngles(math.degrees(math.atan2(y, z)), math.degrees(math.asin(x / length)))
