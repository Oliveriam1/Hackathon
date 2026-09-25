"""Přesná poloha červené tečky vůči zelené (známé) tečce.

Proč: poloha dronu z GPS/EKF během letu pomalu ujíždí (desítky cm až metry
za minuty). Chyba je ale stejná pro všechno, co kamera změří ve stejném
okamžiku. Proto se červená i zelená měří kamerou ve stejné soustavě EKF
a z výsledku se použije jen rozdíl:

    červená_GPS = zelená_GPS + (červená_změřená - zelená_změřená)

Aby se drift mezi oběma měřeními neprojevil, střídají se měření v krátkém
sledu R1 G1 R2 G2 ... a poloha zelené v čase měření červené se lineárně
interpoluje ze sousedních měření zelené (odstraní lineární drift).
Každé měření = medián vzorků z několika snímků při visení v klidu nad bodem,
kamera kolmo dolů. Stejná výška a stejný kurz: zbylé chyby montáže kamery
se v rozdílu odečtou.
"""
import math
import statistics
from dataclasses import dataclass
from .field import GroundPoint
from .geolocation import offset_to_latlon


@dataclass(frozen=True)
class Measurement:
    kind: str            # 'red' | 'green'
    point: GroundPoint   # medián v lokální soustavě EKF (sever/východ od startu)
    time: float          # střední čas vzorků
    spread_m: float      # rozptyl vzorků (medián vzdálenosti od mediánu)
    samples: int


class SampleCollector:
    """Sbírá čerstvé vzorky jednoho bodu; odmítá opakované snímky."""
    def __init__(self, kind, *, needed=15, max_spread=.15):
        self.kind, self.needed, self.max_spread = kind, needed, max_spread
        self.samples = []
        self.last_time = None

    def add(self, estimate):
        if estimate is None or (self.last_time is not None and estimate.sampled_at <= self.last_time):
            return
        values = (estimate.position.north_m, estimate.position.east_m, estimate.sampled_at)
        if not all(math.isfinite(v) for v in values):
            return
        self.last_time = estimate.sampled_at
        self.samples.append(values)

    @property
    def done(self):
        return len(self.samples) >= self.needed

    def result(self):
        if not self.samples:
            return None
        n = statistics.median(s[0] for s in self.samples)
        e = statistics.median(s[1] for s in self.samples)
        spread = statistics.median(math.hypot(s[0]-n, s[1]-e) for s in self.samples)
        return Measurement(self.kind, GroundPoint(n, e), statistics.fmean(s[2] for s in self.samples),
                           spread, len(self.samples))


def green_at(greens, time, *, max_gap=90.):
    """Poloha zelené v daném čase: interpolace mezi sousedními měřeními, jinak nejbližší."""
    usable = sorted((g for g in greens if abs(g.time-time) <= max_gap), key=lambda g: g.time)
    if not usable:
        return None, None
    before = [g for g in usable if g.time <= time]
    after = [g for g in usable if g.time > time]
    if before and after:
        a, b = before[-1], after[0]
        w = (time-a.time)/(b.time-a.time)
        point = GroundPoint(a.point.north_m+w*(b.point.north_m-a.point.north_m),
                            a.point.east_m+w*(b.point.east_m-a.point.east_m))
        return point, 'interpolated'
    nearest = min(usable, key=lambda g: abs(g.time-time))
    return nearest.point, 'nearest'


def solve(measurements, green_lat, green_lon):
    """Výsledná poloha červené z posloupnosti měření. None bez páru červená+zelená."""
    reds = [m for m in measurements if m.kind == 'red']
    greens = [m for m in measurements if m.kind == 'green']
    vectors = []
    for red in reds:
        point, method = green_at(greens, red.time)
        if point is not None:
            vectors.append((red.point.north_m-point.north_m, red.point.east_m-point.east_m, method))
    if not vectors:
        return None
    # Interpolace mezi dvěma zelenými odstraní lineární drift; když existuje, má přednost.
    if any(v[2] == 'interpolated' for v in vectors):
        vectors = [v for v in vectors if v[2] == 'interpolated']
    n = statistics.fmean(v[0] for v in vectors)
    e = statistics.fmean(v[1] for v in vectors)
    spread = max((math.hypot(v[0]-n, v[1]-e) for v in vectors), default=0.)
    lat, lon = offset_to_latlon(green_lat, green_lon, n, e)
    return dict(latitude_deg=lat, longitude_deg=(lon+180) % 360-180, datum='WGS84',
                north_from_green_m=n, east_from_green_m=e, distance_from_green_m=math.hypot(n, e),
                bearing_from_green_deg=math.degrees(math.atan2(e, n)) % 360,
                pairs=len(vectors), pair_spread_m=spread,
                methods=sorted({v[2] for v in vectors}),
                measurements=[dict(kind=m.kind, north_m=m.point.north_m, east_m=m.point.east_m,
                                   time=m.time, spread_m=m.spread_m, samples=m.samples)
                              for m in measurements])
