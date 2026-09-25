"""Datový model lokální mapy. Detekce hranic ani rozhodnutí o průletu zde zatím nejsou."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GroundPoint:
    """Metry od startovní reference: sever a východ, nikoli pixely."""
    north_m: float
    east_m: float


@dataclass(frozen=True)
class FieldMap:
    """Prázdná mapa znamená neznámý prostor, nikoli neomezený povolený prostor.

    verified_boundary musí budoucí mapovač dodat jako uzavřený obvod
    (poslední bod se propojuje s prvním). observed_at je monotónní čas.
    """
    verified_boundary: tuple[GroundPoint, ...] = ()
    observed_at: float | None = None
