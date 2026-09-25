"""Úseky pásky z obrazu -> zakázané čáry v lokální mapě (metry od startu).

Pixelové úsečky z RedWhiteTapeDetector se promítnou na rovnou zem stejnou
geometrií jako tečka (geolocation.py). Opakovaná pozorování stejné čáry se
slučují; čára platí až po min_hits nezávislých snímcích, aby jeden falešný
nález nezablokoval pole. Kvůli časovému sladění volající předává polohu dronu
a úhly kamery platné pro daný snímek.
"""
import math
from dataclasses import replace
from .field import ForbiddenLine, GroundPoint
from .geolocation import _ground_offset


def project_segments(segments_px, pose, gimbal, camera, vehicle, *, image_top='forward',
                     max_off_nadir=60.):
    """Vrátí seznam (GroundPoint, GroundPoint) v lokální mapě; neplatné úseky vynechá."""
    result = []
    for x0, y0, x1, y1 in segments_px:
        ends = []
        for pixel in ((x0, y0), (x1, y1)):
            offset = _ground_offset(pixel, pose, gimbal, camera, image_top)
            if offset is None or offset[2] > max_off_nadir:
                break
            ends.append(GroundPoint(vehicle.north_m+offset[0], vehicle.east_m+offset[1]))
        if len(ends) == 2 and all(math.isfinite(v) for p in ends for v in (p.north_m, p.east_m)):
            result.append(tuple(ends))
    return result


class TapeMapper:
    def __init__(self, *, min_hits=2, merge_distance=1., merge_angle_deg=15., merge_gap=3.,
                 min_length=.4, min_start_distance=1.5, max_lines=24):
        self.min_hits, self.merge_distance = min_hits, merge_distance
        self.merge_cos = math.cos(math.radians(merge_angle_deg))
        self.merge_gap, self.min_length = merge_gap, min_length
        self.min_start_distance, self.max_lines = min_start_distance, max_lines
        self.candidates: list[ForbiddenLine] = []
        self.rejected = 0

    @property
    def lines(self):
        return tuple(c for c in self.candidates if c.hits >= self.min_hits)

    def _merge(self, line, a, b, now):
        ux, uy = line.b.north_m-line.a.north_m, line.b.east_m-line.a.east_m
        length = math.hypot(ux, uy)
        ux, uy = ux/length, uy/length
        vx, vy = b.north_m-a.north_m, b.east_m-a.east_m
        v_length = math.hypot(vx, vy)
        if abs(ux*vx+uy*vy)/v_length < self.merge_cos:
            return None
        o = line.a
        along = lambda p: (p.north_m-o.north_m)*ux+(p.east_m-o.east_m)*uy
        across = lambda p: -(p.north_m-o.north_m)*uy+(p.east_m-o.east_m)*ux
        if max(abs(across(a)), abs(across(b))) > self.merge_distance:
            return None
        s_old = (0., length)
        s_new = sorted((along(a), along(b)))
        if s_new[0] > s_old[1]+self.merge_gap or s_new[1] < s_old[0]-self.merge_gap:
            return None
        # Posun napříč: vážený průměr podle počtu pozorování.
        shift = (across(a)+across(b))/2/(line.hits+1)
        lo, hi = min(s_old[0], s_new[0]), max(s_old[1], s_new[1])
        point = lambda s: GroundPoint(o.north_m+ux*s-uy*shift, o.east_m+uy*s+ux*shift)
        return ForbiddenLine(point(lo), point(hi), now, line.hits+1)

    def update(self, ground_segments, now):
        """Přidá promítnuté úseky jednoho snímku. Každá čára získá nejvýš jeden zásah za snímek."""
        touched = set()
        for a, b in ground_segments:
            line = ForbiddenLine(a, b, now)
            if line.length < self.min_length:
                continue
            # Čára přes start by zakázala vlastní vzletový bod: téměř jistě omyl.
            if abs(line.signed_distance(GroundPoint(0., 0.))) < self.min_start_distance:
                self.rejected += 1
                continue
            for i, old in enumerate(self.candidates):
                merged = self._merge(old, a, b, now)
                if merged is not None:
                    if i in touched:
                        merged = replace(merged, hits=old.hits)
                    self.candidates[i] = merged
                    touched.add(i)
                    break
            else:
                if len(self.candidates) < self.max_lines:
                    self.candidates.append(line)
                    touched.add(len(self.candidates)-1)
        return self.lines
