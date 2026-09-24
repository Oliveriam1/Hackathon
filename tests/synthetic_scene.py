"""Syntetická scéna pro testy: terč na zemi snímaný perspektivní kamerou.

Nejde o náhradu reálné kamery. Ověřuje geometrii (náklon, rotace, výška),
šum, rozmazání a expozici, které na reálné kameře nastanou.
"""
import math

import cv2
import numpy as np

WORLD_SIZE_M = 8.0
WORLD_PX = 2400
M_PER_PX = WORLD_SIZE_M / WORLD_PX


def world_to_texture(x_m, y_m):
    return x_m / M_PER_PX + WORLD_PX / 2, y_m / M_PER_PX + WORLD_PX / 2


def _rectangle(texture, center_m, size_m, color, thickness_m=None):
    (x, y), (w, h) = center_m, size_m
    p1 = np.rint(world_to_texture(x - w/2, y - h/2)).astype(int)
    p2 = np.rint(world_to_texture(x + w/2, y + h/2)).astype(int)
    thickness = -1 if thickness_m is None else max(1, round(thickness_m / M_PER_PX))
    cv2.rectangle(texture, tuple(p1), tuple(p2), color, thickness, cv2.LINE_AA)


def _circle(texture, center_m, radius_m, color, thickness_m=None):
    center = np.rint(world_to_texture(*center_m)).astype(int)
    thickness = -1 if thickness_m is None else max(1, round(thickness_m / M_PER_PX))
    cv2.circle(texture, tuple(center), round(radius_m / M_PER_PX), color, thickness, cv2.LINE_AA)


def build_world(style='sheet', decoys=True, seed=0):
    """Vrátí (svět, samotná tráva). Střed kolečka terče je v bodě (0, 0) m."""
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 1, (WORLD_PX // 40, WORLD_PX // 40, 3)).astype(np.float32)
    low = cv2.resize(small, (WORLD_PX, WORLD_PX), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (WORLD_PX, WORLD_PX, 3)).astype(np.float32), (0, 0), 2)
    grass = np.array([55, 115, 75], np.float32) + low * 14 + fine * 30
    texture = np.clip(grass, 0, 255).astype(np.uint8)
    ground = texture.copy()
    # Pěšina a kameny: přirozené hrany mimo terč.
    cv2.line(texture, world_to_texture_int(-4, 2.6), world_to_texture_int(4, 1.4), (120, 140, 150), 90)
    for _ in range(25):
        x, y = rng.uniform(-3.8, 3.8, 2)
        if abs(x) < 1.0 and abs(y) < 0.8:
            continue
        _circle(texture, (x, y), rng.uniform(0.03, 0.12), tuple(int(v) for v in rng.integers(60, 200, 3)))

    if style == 'sheet':
        # Bílý list 1.0 x 0.7 m, černé kolečko o průměru 0.4 m.
        _rectangle(texture, (0, 0), (1.0, 0.7), (235, 235, 235))
        _circle(texture, (0, 0), 0.2, (25, 25, 25))
    elif style == 'outline':
        # Nakreslený tmavý obdélník a prstenec na světlém podkladu.
        _rectangle(texture, (0, 0), (1.2, 0.9), (200, 205, 200))
        _rectangle(texture, (0, 0), (1.0, 0.7), (30, 30, 30), thickness_m=0.04)
        _circle(texture, (0, 0), 0.22, (30, 30, 30), thickness_m=0.05)
    elif style == 'color':
        # Červené kolečko na bílém listu: v šedi menší kontrast.
        _rectangle(texture, (0, 0), (1.0, 0.7), (225, 230, 230))
        _circle(texture, (0, 0), 0.2, (40, 40, 200))
    elif style == 'small_sheet':
        # Skutečná velikost: kolečko o průměru 200 mm na bílém čtverci 35 x 35 cm.
        _rectangle(texture, (0, 0), (0.35, 0.35), (235, 235, 235))
        _circle(texture, (0, 0), 0.1, (25, 25, 25))
    elif style == 'small_outline':
        # Kolečko 200 mm a čtverec 35 cm nakreslené čarou 15 mm na papíře 45 x 45 cm.
        _rectangle(texture, (0, 0), (0.45, 0.45), (215, 215, 215))
        _rectangle(texture, (0, 0), (0.35, 0.35), (30, 30, 30), thickness_m=0.015)
        _circle(texture, (0, 0), 0.1, (30, 30, 30), thickness_m=0.015)
    else:
        raise ValueError(style)

    if decoys:
        _circle(texture, (-2.0, -1.4), 0.22, (20, 20, 20))                    # samotné kolečko
        _rectangle(texture, (2.0, -1.4), (1.0, 0.7), (235, 235, 235))         # prázdný list
        _rectangle(texture, (-2.0, 1.2), (1.0, 0.7), (235, 235, 235))         # čtverec v listu
        _rectangle(texture, (-2.0, 1.2), (0.35, 0.35), (25, 25, 25))
        _circle(texture, (2.0, 1.1), 0.25, (230, 230, 230), thickness_m=0.05)  # samotný prstenec
    return texture, ground


def world_to_texture_int(x, y):
    return tuple(np.rint(world_to_texture(x, y)).astype(int))


def camera_homography(height, tilt_deg, heading_deg, roll_deg=0.0, look_at=(0.0, 0.0),
                      size=(640, 480), hfov_deg=62.2):
    """Homografie rovina země (m) -> obraz pro dírkovou kameru.

    Kamera ve výšce `height` se dívá na bod `look_at` s náklonem `tilt_deg`
    od svislice (0 = kolmo dolů), směrem `heading_deg`, s rotací `roll_deg`.
    """
    width, image_height = size
    tilt, heading, roll = map(math.radians, (tilt_deg, heading_deg, roll_deg))
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[focal, 0, width/2], [0, focal, image_height/2], [0, 0, 1]])
    direction = np.array([math.cos(heading), math.sin(heading), 0.0])
    look = np.array([look_at[0], look_at[1], 0.0])
    camera = look - direction * height * math.tan(tilt) + np.array([0, 0, height])
    forward = look - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, direction)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    right, down = (right * math.cos(roll) + down * math.sin(roll),
                   -right * math.sin(roll) + down * math.cos(roll))
    R = np.stack([right, down, forward])
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ camera])


def render(world, homography, size=(640, 480), *, blur=0.8, noise=5.0, gain=1.0,
           motion=0, jpeg=80, seed=0):
    """Vykreslí snímek s převzorkováním a degradacemi reálné kamery.

    Mimo modelovaný svět je jen tráva, aby se terč neopakoval.
    """
    texture, ground = world
    scale = np.diag([2.0, 2.0, 1.0])
    to_texture = np.array([[M_PER_PX, 0, -WORLD_PX/2*M_PER_PX],
                           [0, M_PER_PX, -WORLD_PX/2*M_PER_PX], [0, 0, 1]])
    matrix, big_size = scale @ homography @ to_texture, (size[0]*2, size[1]*2)
    big = cv2.warpPerspective(texture, matrix, big_size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), matrix, big_size,
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    outside = cv2.warpPerspective(ground, matrix, big_size, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
    big = np.where(inside[..., None] > 0, big, outside)
    frame = cv2.resize(big, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    if motion:
        kernel = np.zeros((motion, motion), np.float32)
        kernel[motion // 2, :] = 1 / motion
        frame = cv2.filter2D(frame, -1, kernel)
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    rng = np.random.default_rng(seed)
    frame = frame * gain + rng.normal(0, noise, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if jpeg:
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return frame


def project(homography, point_m):
    point = homography @ np.array([point_m[0], point_m[1], 1.0])
    return point[:2] / point[2]


def flight(frames=60, seed=0):
    """Přiblížení z šikmého pohledu nad terč: náklon 60° -> 0°, výška 9 -> 4 m.

    Vrací seznam (homografie, degradace). Každý osmý snímek je výrazně
    rozmazaný pohybem nebo přeexponovaný, jako při prudkém manévru.
    """
    rng = np.random.default_rng(seed)
    result = []
    for index in range(frames):
        t = index / max(1, frames - 1)
        tilt = 60 * (1 - t)
        height = 9 - 5 * t
        heading = 30 + 120 * t
        roll = 15 * math.sin(t * 6)
        look = (0.6 * math.sin(t * 5) * (1 - t) + rng.normal(0, 0.05),
                0.4 * math.cos(t * 4) * (1 - t) + rng.normal(0, 0.05))
        degrade = dict(blur=rng.uniform(0.5, 1.3), noise=rng.uniform(3, 8),
                       gain=rng.uniform(0.75, 1.15), seed=int(rng.integers(1 << 30)))
        if index % 8 == 7:
            degrade.update(motion=9) if index % 16 == 7 else degrade.update(gain=1.6)
        result.append((camera_homography(height, tilt, heading, roll, look), degrade))
    return result
