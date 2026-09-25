"""Datový model lokální mapy: ověřený obvod a zakázané čáry z pásky."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GroundPoint:
    """Metry od startovní reference: sever a východ, nikoli pixely."""
    north_m: float
    east_m: float


@dataclass(frozen=True)
class ForbiddenLine:
    """Čára pásky na zemi. Zakázaná je celá polorovina za ní.

    Povolená strana je ta, na které leží startovní reference (0, 0): start je
    podle zadání uvnitř vyhrazené oblasti. Čára se chová jako nekonečná přímka,
    což je u rovných úseků pásky konzervativní (neumožní ji obletět).
    """
    a: GroundPoint
    b: GroundPoint
    observed_at: float
    hits: int = 1

    @property
    def length(self):
        return math.hypot(self.b.north_m-self.a.north_m, self.b.east_m-self.a.east_m)

    def _raw(self, point):
        dn, de = self.b.north_m-self.a.north_m, self.b.east_m-self.a.east_m
        return (dn*(point.east_m-self.a.east_m)-de*(point.north_m-self.a.north_m))/self.length

    def signed_distance(self, point):
        """Kladně na straně startu, záporně za páskou (metry)."""
        if self.length < 1e-6:
            return math.inf
        sign = 1. if self._raw(GroundPoint(0., 0.)) >= 0 else -1.
        return sign*self._raw(point)

    def inward(self):
        """Jednotkový vektor (sever, východ) kolmo od pásky směrem ke startu."""
        dn, de = self.b.north_m-self.a.north_m, self.b.east_m-self.a.east_m
        sign = 1. if self._raw(GroundPoint(0., 0.)) >= 0 else -1.
        return -de/self.length*sign, dn/self.length*sign


@dataclass(frozen=True)
class FieldMap:
    """Prázdná mapa znamená neznámý prostor, nikoli neomezený povolený prostor.

    verified_boundary: uzavřený obvod (např. změřené GPS rohy pole), poslední bod
    se propojuje s prvním. observed_at je monotónní čas. forbidden_lines jsou
    potvrzené úseky pásky; neexpirují, páska se nehýbe.
    """
    verified_boundary: tuple[GroundPoint, ...] = ()
    observed_at: float | None = None
    forbidden_lines: tuple[ForbiddenLine, ...] = ()

    def line_clearance(self, point):
        """Nejmenší vzdálenost od pásky na povolené straně (inf bez pásky)."""
        return min((line.signed_distance(point) for line in self.forbidden_lines), default=math.inf)
