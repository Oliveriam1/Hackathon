"""Rozhraní budoucího plánovače. Neznámá mapa nesmí znamenat volný let."""
from typing import Protocol
from .field import FieldMap, GroundPoint


class SearchPlanner(Protocol):
    def next_point(self, field: FieldMap, position: GroundPoint) -> GroundPoint | None:
        """Navrhne další bod; None, pokud nelze bezpečný bod určit. Neposílá povely."""
        ...
