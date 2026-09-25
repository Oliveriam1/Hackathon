"""Podprogram: souřadnice tečky z polohy dronu a úhlů kamery (trojúhelník).

Vstup:
  DronePosition – GPS dronu, výška kamery nad zemí, kurz přídě (od severu po
                  směru hodinových ručiček), volitelně náklon dronu,
  úhly kamery   – right / forward z míření (AimResult), 0/0 = kolmo dolů.
Výstup:
  TargetPosition – GPS tečky, posun od dronu v metrech, vzdálenost, azimut
                   a odhad chyby.

Předpoklad: zem je rovná a tečka leží ve výšce, od které se měří height_m.

Příklad:
    from src.locate import DronePosition, locate_target
    drone = DronePosition(50.0875123, 14.4213456, height_m=5.0, heading_deg=0.0)
    target = locate_target(drone, right_deg=10.0, forward_deg=-5.0)
    print(target.latitude_deg, target.longitude_deg)
"""
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from .geolocation import DronePose, GimbalAngles, body_in_ned, camera_in_body, offset_to_latlon

MAX_OFF_NADIR_DEG = 75.0   # šikmější paprsek protne zem daleko a s obrovskou chybou


@dataclass(frozen=True)
class DronePosition:
    latitude_deg: float
    longitude_deg: float
    height_m: float          # výška kamery nad zemí, na které leží tečka
    heading_deg: float       # kurz přídě od severu, po směru hodinových ručiček
    roll_deg: float = 0.0    # kladně = pravé rameno dolů
    pitch_deg: float = 0.0   # kladně = příď nahoru

    def __post_init__(self):
        values = (self.latitude_deg, self.longitude_deg, self.height_m, self.heading_deg,
                  self.roll_deg, self.pitch_deg)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('Poloha dronu obsahuje neplatné číslo.')
        if not -89.9 < self.latitude_deg < 89.9 or not -180 <= self.longitude_deg <= 180:
            raise ValueError('Neplatné GPS souřadnice dronu.')
        if not 0 < self.height_m <= 500:
            raise ValueError('Výška dronu musí být v rozsahu (0, 500] m.')
        if abs(self.roll_deg) > 45 or abs(self.pitch_deg) > 45:
            raise ValueError('Náklon dronu nad 45° není podporován.')

    @classmethod
    def from_json(cls, path):
        """{"latitude_deg": .., "longitude_deg": .., "height_m": .., "heading_deg": .., "roll_deg": 0, "pitch_deg": 0}"""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(float(data['latitude_deg']), float(data['longitude_deg']), float(data['height_m']),
                   float(data['heading_deg']), float(data.get('roll_deg', 0.)), float(data.get('pitch_deg', 0.)))


@dataclass(frozen=True)
class Uncertainty:
    """Směrodatné odchylky vstupů (1 sigma) pro odhad chyby výsledku."""
    camera_deg: float = 1.0      # úhel kamery (serva po kalibraci + míření)
    attitude_deg: float = 1.0    # náklon dronu (roll/pitch)
    heading_deg: float = 2.0     # kurz dronu
    height_m: float = 0.1        # výška kamery nad zemí
    drone_position_m: float = 0.0  # chyba polohy dronu (dodává navigace)


@dataclass(frozen=True)
class TargetPosition:
    latitude_deg: float
    longitude_deg: float
    north_m: float           # posun tečky od dronu na sever
    east_m: float            # posun tečky od dronu na východ
    distance_m: float        # vodorovná vzdálenost od dronu
    bearing_deg: float       # azimut od dronu, od severu
    off_nadir_deg: float     # odklon paprsku od svislice
    error_m: float           # odhad vodorovné chyby (1 sigma)

    def as_dict(self):
        return dict(latitude_deg=self.latitude_deg, longitude_deg=self.longitude_deg, datum='WGS84',
                    north_m=self.north_m, east_m=self.east_m, distance_m=self.distance_m,
                    bearing_deg=self.bearing_deg, off_nadir_deg=self.off_nadir_deg, error_m=self.error_m)


def _ground_offset(drone, angles):
    """(sever, východ, odklon) průsečíku osy kamery se zemí, nebo None."""
    pose = DronePose(0., 0., drone.height_m, drone.roll_deg, drone.pitch_deg, drone.heading_deg)
    # Osa kamery je její osa z; natočení obrazu kolem osy na výsledek nemá vliv.
    ray = body_in_ned(pose) @ camera_in_body(angles) @ np.array([0., 0., 1.])
    ray /= np.linalg.norm(ray)
    off_nadir = math.degrees(math.acos(max(-1., min(1., ray[2]))))
    if off_nadir > MAX_OFF_NADIR_DEG:
        return None
    distance = drone.height_m/ray[2]
    return distance*ray[0], distance*ray[1], off_nadir


def locate_target(drone, right_deg, forward_deg, *, uncertainty=Uncertainty()):
    """Souřadnice tečky, na kterou kamera míří. None, když paprsek zem neprotne rozumně blízko."""
    if not all(math.isfinite(v) for v in (right_deg, forward_deg)):
        raise ValueError('Úhly kamery musí být konečná čísla.')
    angles = GimbalAngles(right_deg, forward_deg)
    offset = _ground_offset(drone, angles)
    if offset is None:
        return None
    north, east, off_nadir = offset
    # Chyba: každý vstup posuneme o jeho sigma a sečteme čtverce posunů výsledku.
    variants = [(drone, GimbalAngles(right_deg+uncertainty.camera_deg, forward_deg)),
                (drone, GimbalAngles(right_deg, forward_deg+uncertainty.camera_deg)),
                (replace(drone, roll_deg=drone.roll_deg+uncertainty.attitude_deg), angles),
                (replace(drone, pitch_deg=drone.pitch_deg+uncertainty.attitude_deg), angles),
                (replace(drone, heading_deg=drone.heading_deg+uncertainty.heading_deg), angles),
                (replace(drone, height_m=drone.height_m+uncertainty.height_m), angles)]
    variance = uncertainty.drone_position_m**2
    for variant in variants:
        moved = _ground_offset(*variant)
        variance += math.inf if moved is None else (moved[0]-north)**2+(moved[1]-east)**2
    lat, lon = offset_to_latlon(drone.latitude_deg, drone.longitude_deg, north, east)
    return TargetPosition(lat, (lon+180) % 360-180, north, east, math.hypot(north, east),
                          math.degrees(math.atan2(east, north)) % 360, off_nadir, math.sqrt(variance))
