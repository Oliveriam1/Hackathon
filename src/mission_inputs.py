"""Převod vizuálního kontraktu do lokální reference mise; bez letových povelů."""
import math
from .approach import TargetEstimate
from .field import GroundPoint
from .geolocation import offset_to_latlon


def target_from_drone_data(data, start, *, sampled_at, now):
    """sampled_at dodává producent snímku, nesmí se obnovovat čtením starého JSON.

    Pouze převod. Schválení kvality mapy, serv a synchronizace patří do backendu.
    """
    if (not all(math.isfinite(v) for v in (sampled_at, now)) or not 0 <= now-sampled_at <= .25 or
            not data.get('live_control_input_valid') or not data.get('measurement_valid') or
            data.get('input_source') != 'camera' or data.get('target_id') is None):
        return None
    fix = data.get('world_position')
    if not fix or fix.get('datum') != 'WGS84' or data.get('world_position_status') != 'ESTIMATED':
        return None
    lat, lon, error = (fix.get(k) for k in ('latitude_deg', 'longitude_deg', 'horizontal_error_estimate_m'))
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lat, lon, error)):
        return None
    if not -89.9 < lat < 89.9 or not -180 <= lon <= 180 or not 0 <= error <= 2:
        return None
    if not -89.9 < start.latitude_deg < 89.9:
        return None
    north_unit, _ = offset_to_latlon(start.latitude_deg, start.longitude_deg, 1., 0.)
    _, east_unit = offset_to_latlon(start.latitude_deg, start.longitude_deg, 0., 1.)
    north = (lat-start.latitude_deg)/(north_unit-start.latitude_deg)
    east = ((lon-start.longitude_deg+180)%360-180)/(east_unit-start.longitude_deg)
    if math.hypot(north, east) > 1000:
        return None  # Tento lokální model není plánovač dálkového letu.
    return TargetEstimate(GroundPoint(north, east), sampled_at, str(data['target_id']), error)
