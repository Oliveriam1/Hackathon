"""Vizuální zámek a chyba centrování. Souřadnice obrazu, nikoli letové povely."""
import math


class TargetLock:
    def __init__(self, max_age_s=0.25, settle_s=0.5, enter_tolerance=0.03, exit_tolerance=0.05):
        if not (0 < max_age_s and 0 < settle_s and 0 < enter_tolerance < exit_tolerance < 1):
            raise ValueError('Neplatné parametry zámku.')
        self.max_age_s, self.settle_s = max_age_s, settle_s
        self.enter_tolerance, self.exit_tolerance = enter_tolerance, exit_tolerance
        self._centered_since = None
        self._last_sample = None
        self._last_center = None

    def reset(self):
        self._centered_since = None
        self._last_sample = None
        self._last_center = None

    def update(self, observation, *, sample_time, now):
        finite_time = math.isfinite(sample_time) and math.isfinite(now)
        result = {'state': 'SEARCHING', 'locked': False, 'image_centered': False,
                  'error_normalized': None, 'age_s': max(0.0, now-sample_time) if finite_time else None,
                  'coordinate_frame': 'image_right_down', 'flight_command': None}
        previous_time = self._last_sample
        if (not math.isfinite(sample_time) or not math.isfinite(now)
                or sample_time > now or now-sample_time > self.max_age_s
                or (previous_time is not None and sample_time <= previous_time)):
            self.reset()
            self._last_sample = previous_time
            if finite_time and sample_time <= now:
                self._last_sample = max(previous_time, sample_time) if previous_time is not None else sample_time
            result['state'] = 'STALE_OR_REPEATED'
            return result
        if previous_time is not None and sample_time-previous_time > self.max_age_s:
            self._centered_since = None
        self._last_sample = sample_time
        red = getattr(observation, 'red', None)
        if red and red.get('tracking', {}).get('state') in ('AMBIGUOUS', 'STALE_OR_REPEATED'):
            self._centered_since = self._last_center = None
            result['state'] = red['tracking']['state']
            return result
        measurement = observation.measurement
        if not observation.confirmed or not observation.measured or measurement is None:
            self._centered_since = None
            self._last_center = None
            result['state'] = 'LOST' if observation.confirmed else 'SEARCHING'
            return result
        width, height = observation.frame_size
        ex, ey = (measurement.x-width/2)/(width/2), (measurement.y-height/2)/(height/2)
        if not all(math.isfinite(v) for v in (ex, ey)) or abs(ex) > 1 or abs(ey) > 1:
            self.reset()
            result['state'] = 'INVALID_MEASUREMENT'
            return result
        if self._last_center is not None and math.dist(self._last_center, (ex, ey)) > .15:
            self._centered_since = None
        self._last_center = (ex, ey)
        error = max(abs(ex), abs(ey))
        if error > self.exit_tolerance:
            self._centered_since = None
        elif error <= self.enter_tolerance and self._centered_since is None:
            self._centered_since = sample_time
        centered = self._centered_since is not None and sample_time-self._centered_since >= self.settle_s
        result.update(state='IMAGE_CENTERED' if centered else 'LOCKED', locked=True,
                      image_centered=centered, error_normalized={'x': ex, 'y': ey})
        return result
