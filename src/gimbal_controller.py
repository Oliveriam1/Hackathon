"""Časový P regulátor centrování kamery; bez přímého přístupu k GPIO."""
import math
from .geolocation import GimbalAngles, centering_angles


class GimbalController:
    def __init__(self, camera, *, image_top='forward', gain=2.0, max_speed=15.,
                 deadband=.4, max_age=.25, x_limits=(-60., 60.), y_limits=(-45., 45.)):
        self.camera, self.image_top = camera, image_top
        self.gain, self.max_speed, self.deadband, self.max_age = gain, max_speed, deadband, max_age
        self.x_limits, self.y_limits = x_limits, y_limits
        self.last_sample = self.last_time = None
        self.status = 'WAITING'

    def update(self, observation, angles, *, sample_time, now):
        self.status = 'HOLD'
        dt = .02 if self.last_time is None else max(0., min(.1, now-self.last_time))
        self.last_time = now
        fresh = (math.isfinite(now) and math.isfinite(sample_time) and
                 0 <= now-sample_time < self.max_age and
                 (self.last_sample is None or sample_time > self.last_sample))
        if not fresh:
            self.status = 'STALE_OR_REPEATED'
            return None
        self.last_sample = sample_time
        if (not observation.confirmed or not observation.measured or observation.measurement is None
                or (getattr(observation, 'red', None) or {}).get('tracking', {}).get('state') == 'AMBIGUOUS'):
            return None
        if tuple(observation.frame_size) != tuple(self.camera.size):
            raise ValueError('Rozlišení snímku neodpovídá modelu regulátoru.')
        point = observation.measurement
        width, height = observation.frame_size
        if not (math.isfinite(point.x) and math.isfinite(point.y) and
                0 <= point.x < width and 0 <= point.y < height):
            self.status = 'INVALID_MEASUREMENT'
            return None
        desired = centering_angles((point.x, point.y), angles, self.camera, self.image_top)
        errors = (desired.right-angles.right, desired.forward-angles.forward)
        step = lambda e: 0. if abs(e) <= self.deadband else max(-self.max_speed, min(self.max_speed, self.gain*e))*dt
        x = max(self.x_limits[0], min(self.x_limits[1], angles.right+step(errors[0])))
        y = max(self.y_limits[0], min(self.y_limits[1], angles.forward+step(errors[1])))
        if x == angles.right and y == angles.forward:
            self.status = 'CENTERED' if max(map(abs, errors)) <= self.deadband else 'LIMIT'
            return None
        self.status = 'TRACKING'
        return GimbalAngles(x, y)


class GimbalScanner:
    """Prohledávání okolí servy, když tečka není v záběru.

    Kamera postupně míří na body mřížky (nejdřív kolmo dolů, pak prstenec kolem)
    a na každém chvíli počká, aby detektor stihl tečku potvrdit ve více snímcích.
    Rozestup 25° je menší než zorné pole OV5647 (54° x 41°), záběry se překrývají.
    """
    PATTERN = ((0., 0.), (25., 0.), (25., 25.), (0., 25.), (-25., 25.), (-25., 0.), (-25., -25.),
               (0., -25.), (25., -25.), (50., 0.), (50., 40.), (0., 40.), (-50., 40.), (-50., 0.),
               (-50., -40.), (0., -40.), (50., -40.))

    def __init__(self, *, max_speed=20., dwell=.8, x_limits=(-60., 60.), y_limits=(-45., 45.)):
        self.max_speed, self.dwell = max_speed, dwell
        self.x_limits, self.y_limits = x_limits, y_limits
        self.reset()

    def reset(self):
        self.index = 0
        self.arrived_at = None
        self.last_time = None

    def update(self, angles, now):
        """Další krok k aktuálnímu bodu mřížky, nebo None (kamera na bodě a čeká)."""
        dt = 0. if self.last_time is None else max(0., min(.1, now-self.last_time))
        self.last_time = now
        tx, ty = self.PATTERN[self.index % len(self.PATTERN)]
        tx = max(self.x_limits[0], min(self.x_limits[1], tx))
        ty = max(self.y_limits[0], min(self.y_limits[1], ty))
        dx, dy = tx-angles.right, ty-angles.forward
        distance = math.hypot(dx, dy)
        if distance < .5:
            if self.arrived_at is None:
                self.arrived_at = now
            if now-self.arrived_at >= self.dwell:
                self.index += 1
                self.arrived_at = None
            return None
        self.arrived_at = None
        step = min(distance, self.max_speed*dt)
        return GimbalAngles(angles.right+dx/distance*step, angles.forward+dy/distance*step)
