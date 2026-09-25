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
                or (observation.red or {}).get('tracking', {}).get('state') == 'AMBIGUOUS'):
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
