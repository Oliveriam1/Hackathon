"""Odhad polohy cíle nad explicitně zadanou rovinou; bez letových povelů."""
import math
from .geolocation import CameraModel, DronePose, locate
from .telemetry import MAX_AGE


class TargetLocator:
    def __init__(self, config):
        self.config = config
        self.model = CameraModel.load(config.camera_calibration) if config.camera_calibration else None

    def estimate(self, record, angles, angle_basis):
        if not self.config.locate_target:
            return None, 'DISABLED'
        if record['source'] != 'camera':
            return None, 'NON_LIVE_SOURCE'
        lock = record['visual_lock']
        if not record['valid_pixel_position'] or not lock['locked'] or not 0 <= lock['age_s'] < .25:
            return None, 'NO_FRESH_TARGET'
        if angles is None:
            return None, 'CAMERA_ANGLES_UNKNOWN'
        if self.config.ground_relative_alt is None:
            return None, 'GROUND_PLANE_UNKNOWN'
        telemetry = record.get('telemetry')
        if telemetry is None:
            return None, 'TELEMETRY_MISSING'
        if telemetry.get('error'):
            return None, 'TELEMETRY_ERROR'
        messages = telemetry.get('messages', {})
        for name, limit in MAX_AGE.items():
            item = messages.get(name)
            if item is None:
                return None, 'TELEMETRY_MISSING'
            if not math.isfinite(item['age_s']) or not 0 <= item['age_s'] <= limit:
                return None, 'TELEMETRY_STALE'
        if messages['GPS_RAW_INT']['values']['fix_type'] < 3:
            return None, 'GPS_FIX_INVALID'
        # Hlídání přibližného časového souběhu; není synchronizace expozice.
        for name in ('GLOBAL_POSITION_INT', 'ATTITUDE'):
            if abs(messages[name]['age_s']-lock['age_s']) > .2:
                return None, 'TELEMETRY_TIME_MISMATCH'
        p = messages['GLOBAL_POSITION_INT']['values']
        a = messages['ATTITUDE']['values']
        height = p['relative_alt_m'] - self.config.ground_relative_alt
        values = (p['latitude_deg'], p['longitude_deg'], height, a['roll_deg'], a['pitch_deg'],
                  a['yaw_deg'], angles.right, angles.forward)
        if (not all(math.isfinite(v) for v in values) or not -89.9 < values[0] < 89.9
                or not -180 <= values[1] <= 180 or height <= 0):
            return None, 'POSE_INVALID'
        size = tuple(record['frame_size_px'])
        if self.model is None:
            width, img_height = size
            hfov = self.config.hfov_deg
            vfov = math.degrees(2*math.atan(img_height/width*math.tan(math.radians(hfov/2))))
            model = CameraModel.from_fov(size, (hfov, vfov))
        else:
            model = self.model
        if tuple(model.size) != size:
            return None, 'CALIBRATION_SIZE_MISMATCH'
        point = record['measurement_px']
        if point is None or not (0 <= point['x'] < size[0] and 0 <= point['y'] < size[1]):
            return None, 'PIXEL_INVALID'
        pose = DronePose(*values[:6])
        fix = locate((point['x'], point['y']), pose, angles, model, frame_size=size,
                     image_top=self.config.image_top)
        if fix is None:
            return None, 'RAY_OUTSIDE_GROUND_RANGE'
        if not all(math.isfinite(v) for v in (fix.lat, fix.lon, fix.north, fix.east, fix.error)):
            return None, 'UNCERTAINTY_UNBOUNDED'
        return dict(latitude_deg=fix.lat, longitude_deg=(fix.lon+180)%360-180,
                    offset_north_m=fix.north, offset_east_m=fix.east,
                    distance_m=fix.distance, bearing_deg=fix.bearing,
                    off_nadir_deg=fix.off_nadir, height_above_target_plane_m=height,
                    horizontal_error_estimate_m=fix.error, datum='WGS84', quality='ESTIMATED',
                    camera_model='calibrated' if self.model else 'nominal_fov',
                    camera_angles_deg={'right': angles.right, 'forward': angles.forward},
                    angle_basis=angle_basis, ground_relative_alt_m=self.config.ground_relative_alt,
                    time_basis='host_receive_not_exposure_synchronized',
                    uncertainty_basis='default_input_sigmas_not_field_validated'), 'ESTIMATED'
