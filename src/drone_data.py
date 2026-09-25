"""Kontrakt vizuálních dat pro budoucí řídicí část. Žádné letové povely."""


def drone_record(record):
    lock = record['visual_lock']
    valid = bool(record['valid_pixel_position'] and lock['locked'])
    tracking = record.get('tracking') or {}
    point = record['measurement_px'] if valid else None
    return {
        'schema_version': 1,
        'message_type': 'vision_target',
        'sequence': record['sequence'],
        'received_at_unix_s': record['received_at_unix_s'],
        'timestamp_basis': record['timestamp_basis'],
        'input_source': record['source'],
        'state': lock['state'],
        'measurement_valid': valid,
        'measurement_age_ms': round(lock['age_s']*1000, 3) if lock['age_s'] is not None else None,
        'target_id': tracking.get('track_id'),
        'candidate_count': record['candidate_count'],
        'frame_size_px': record['frame_size_px'],
        'target_px': {'x': point['x'], 'y': point['y']} if point is not None else None,
        'camera_error_normalized': lock['error_normalized'] if valid else None,
        'coordinate_frame': 'image_right_down',
        'image_centered': bool(valid and lock['image_centered']),
        'world_position': record['geolocation'] if valid else None,
        'flight_ready': False,
        'flight_command': None,
    }
