"""Rozhraní budoucího plánovače. Neznámá mapa nesmí znamenat volný let."""
from typing import Protocol
from .field import FieldMap, GroundPoint
import math


class SearchPlanner(Protocol):
    def next_point(self, field: FieldMap, position: GroundPoint) -> GroundPoint | None:
        """Navrhne další bod; None, pokud nelze bezpečný bod určit. Neposílá povely."""
        ...


class RectangleSweep:
    """Řádky uvnitř explicitního simulačního obdélníku; žádný odhad hranic."""
    def __init__(self, bounds, *, spacing=4., margin=1.):
        if (len(bounds) != 4 or not all(math.isfinite(v) for v in (*bounds, spacing, margin))
                or spacing <= 0 or margin < .5):
            raise ValueError('Neplatná hranice, rozestup nebo rezerva hledání.')
        n0, n1, e0, e1 = bounds
        if n1-n0 <= 2*margin or e1-e0 <= 2*margin:
            raise ValueError('Pole je menší než rezerva u hranic.')
        count = math.ceil((e1-e0-2*margin)/spacing)
        if count > 1000:
            raise ValueError('Příliš mnoho řádků hledání.')
        self.bounds = tuple(bounds)
        self.points = tuple(
            GroundPoint(n, e0+margin+i*(e1-e0-2*margin)/count)
            for i in range(count+1)
            for n in ((n0+margin, n1-margin) if i % 2 == 0 else (n1-margin, n0+margin)))
