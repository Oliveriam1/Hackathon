"""Kontrakt vizuálních dat pro budoucí řídicí část. Žádné letové povely."""
import math


def drone_record(record, *, session_id=None, max_age_s=0.25):
    lock = record['visual_lock']
    age = lock['age_s']
    fresh = age is not None and math.isfinite(age) and 0 <= age < max_age_s
    width, height = record['frame_size_px']
    point = record['measurement_px']
    coordinates_ok = (point is not None and width > 0 and height > 0
                      and all(math.isfinite(point[k]) for k in ('x', 'y'))
                      and 0 <= point['x'] < width and 0 <= point['y'] < height)
    valid = bool(record['valid_pixel_position'] and lock['locked'] and fresh and coordinates_ok)
    tracking = record.get('tracking') or {}
    point = record['measurement_px'] if valid else None
    reason = None if valid else ('STALE_OR_REPEATED' if not fresh else
                                lock['state'] if not lock['locked'] else 'INVALID_MEASUREMENT')
    live = record['source'] == 'camera'
    return {
        'schema_version': 1,
        'message_type': 'vision_target',
        'session_id': session_id,
        'sequence': record['sequence'],
        'received_at_unix_s': record['received_at_unix_s'],
        'timestamp_basis': record['timestamp_basis'],
        'input_source': record['source'],
        'state': lock['state'],
        'tracking_state': tracking.get('state'),
        'measurement_valid': valid,
        'invalid_reason': reason,
        'measurement_age_ms': round(age*1000, 3) if age is not None and math.isfinite(age) else None,
        'max_measurement_age_ms': max_age_s*1000,
        'remaining_validity_ms': max(0., (max_age_s-age)*1000) if valid else 0.,
        'live_control_input_valid': bool(valid and live),
        'target_id': tracking.get('track_id'),
        'candidate_count': record['candidate_count'],
        'frame_size_px': record['frame_size_px'],
        'target_px': {'x': point['x'], 'y': point['y']} if point is not None else None,
        'camera_error_px': {'x': point['x']-width/2, 'y': point['y']-height/2} if valid else None,
        'camera_error_normalized': {'x': (point['x']-width/2)/(width/2),
                                    'y': (point['y']-height/2)/(height/2)} if valid else None,
        'coordinate_frame': 'image_right_down',
        'image_centered': bool(valid and lock['image_centered']),
        'world_position': record['geolocation'] if valid else None,
        'flight_ready': False,
        'flight_command': None,
        'gimbal': record.get('gimbal', {'enabled': False, 'state': 'DISABLED'}),
    }
