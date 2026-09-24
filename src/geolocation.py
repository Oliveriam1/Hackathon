"""Poloha kolečka na zemi (GPS) z pixelu, úhlů závěsu kamery a polohy dronu.

Předpoklady: rovný terén ve výšce místa startu, výška dronu je ArduPilot
relative_alt (nad místem startu). Řetězec rotací:
sever/východ/dolů <- tělo dronu (yaw, pitch, roll) <- vnější servo (osa x)
<- vnitřní servo (osa y) <- kamera. Při úhlech závěsu 0/0 míří kamera kolmo dolů.
"""
from dataclasses import dataclass, replace
import json
import math

import cv2
import numpy as np

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
MAX_OFF_NADIR = 80.0  # téměř vodorovný paprsek zem neprotne v rozumné vzdálenosti
# OV5647 (2592 x 1944 px po 1.4 um) s objektivem 3.6 mm, režim s celým zorným polem.
NOMINAL_FOV = (53.5, 41.4)


@dataclass(frozen=True)
class DronePose:
    lat: float          # stupně WGS84
    lon: float
    height: float       # m nad místem startu (relative_alt)
    roll: float = 0.0   # stupně, kladně = pravé rameno dolů
    pitch: float = 0.0  # stupně, kladně = příď nahoru
    yaw: float = 0.0    # stupně od severu po směru hodinových ručiček


@dataclass(frozen=True)
class GimbalAngles:
    """Skutečné úhly kamery vůči tělu dronu, 0/0 = kolmo dolů."""
    right: float = 0.0    # vnější servo, osa x: kladně = kamera kouká doprava
    forward: float = 0.0  # vnitřní servo, osa y: kladně = kamera kouká dopředu


@dataclass(frozen=True)
class ServoAxis:
    """Příkaz serva -> skutečný úhel: znaménko * měřítko * příkaz + posun.

    MG996R nemá přesný úhel ani nulu, měřítko a posun je nutné změřit (README).
    """
    sign: int = 1
    scale: float = 1.0
    offset: float = 0.0

    def angle(self, command):
        return self.sign * self.scale * command + self.offset

    def command(self, angle):
        return (angle - self.offset) / (self.sign * self.scale)


def gimbal_from_servos(x_command, y_command, x_axis=ServoAxis(), y_axis=ServoAxis()):
    return GimbalAngles(x_axis.angle(x_command), y_axis.angle(y_command))


@dataclass(frozen=True)
class Uncertainty:
    """Směrodatné odchylky vstupů (1 sigma) pro odhad chyby polohy."""
    pixel: float = 1.5     # px, střed kolečka z detektoru
    gimbal: float = 2.0    # °, MG996R po kalibraci; bez ní klidně 5°
    attitude: float = 1.0  # °, roll a pitch z EKF ArduPilotu
    yaw: float = 3.0       # °, kompas
    height: float = 1.0    # m, relative_alt
    gps: float = 1.5       # m, vodorovná poloha dronu bez RTK


@dataclass(frozen=True)
class TargetFix:
    lat: float
    lon: float
    north: float      # m od dronu na sever
    east: float       # m od dronu na východ
    distance: float   # vodorovná vzdálenost od dronu, m
    bearing: float    # směr od dronu, stupně od severu
    off_nadir: float  # odklon paprsku od svislice, stupně
    error: float      # odhad vodorovné chyby polohy (1 sigma), m


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


def _ground_offset(pixel, pose, gimbal, camera, image_top):
    ray = body_in_ned(pose) @ camera_in_body(gimbal, image_top) @ camera.ray(*pixel)
    ray /= np.linalg.norm(ray)
    off_nadir = math.degrees(math.acos(min(1.0, max(-1.0, ray[2]))))
    if pose.height <= 0 or off_nadir > MAX_OFF_NADIR:
        return None
    distance = pose.height / ray[2]
    return distance * ray[0], distance * ray[1], off_nadir


def _error(pixel, pose, gimbal, camera, image_top, sigma, north, east):
    # Každý vstup posuneme o jeho sigma a sečteme čtverce posunů výsledku.
    variants = [((pixel[0] + sigma.pixel, pixel[1]), pose, gimbal),
                ((pixel[0], pixel[1] + sigma.pixel), pose, gimbal),
                (pixel, pose, replace(gimbal, right=gimbal.right + sigma.gimbal)),
                (pixel, pose, replace(gimbal, forward=gimbal.forward + sigma.gimbal)),
                (pixel, replace(pose, roll=pose.roll + sigma.attitude), gimbal),
                (pixel, replace(pose, pitch=pose.pitch + sigma.attitude), gimbal),
                (pixel, replace(pose, yaw=pose.yaw + sigma.yaw), gimbal),
                (pixel, replace(pose, height=pose.height + sigma.height), gimbal)]
    variance = sigma.gps**2
    for variant in variants:
        offset = _ground_offset(*variant, camera, image_top)
        if offset is None:
            return math.inf
        variance += (offset[0] - north)**2 + (offset[1] - east)**2
    return math.sqrt(variance)


def locate(pixel, pose, gimbal, camera, *, frame_size=None, image_top='forward', uncertainty=Uncertainty()):
    """Poloha bodu na zemi viděného v pixelu (u, v), nebo None nad horizontem."""
    if frame_size is not None and tuple(frame_size) != tuple(camera.size):
        raise ValueError(f'Kalibrace platí pro {camera.size}, snímek má {tuple(frame_size)}.')
    offset = _ground_offset(pixel, pose, gimbal, camera, image_top)
    if offset is None:
        return None
    north, east, off_nadir = offset
    lat, lon = offset_to_latlon(pose.lat, pose.lon, north, east)
    error = _error(pixel, pose, gimbal, camera, image_top, uncertainty, north, east)
    return TargetFix(lat, lon, north, east, math.hypot(north, east),
                     math.degrees(math.atan2(east, north)) % 360, off_nadir, error)


def combine(fixes):
    """Vážený průměr měření (váha 1/chyba^2) -> (lat, lon, chyba).

    Chyby jsou z velké části společné (kompas, serva, GPS dronu), průměr je proto
    nezmenší pod chybu nejlepšího měření. Měření nad kolečkem převáží šikmá.
    """
    fixes = [fix for fix in fixes if math.isfinite(fix.error)]
    if not fixes:
        return None
    weights = np.array([1 / fix.error**2 for fix in fixes])
    lat = float(np.dot(weights, [fix.lat for fix in fixes]) / weights.sum())
    lon = float(np.dot(weights, [fix.lon for fix in fixes]) / weights.sum())
    return lat, lon, min(fix.error for fix in fixes)


def centering_angles(pixel, gimbal, camera, image_top='forward'):
    """Úhly závěsu, při kterých se bod z pixelu dostane do středu obrazu."""
    x, y, z = camera_in_body(gimbal, image_top) @ camera.ray(*pixel)
    length = math.sqrt(x*x + y*y + z*z)
    # Osa kamery v těle dronu je (sin f, sin r * cos f, cos r * cos f).
    return GimbalAngles(math.degrees(math.atan2(y, z)), math.degrees(math.asin(x / length)))
