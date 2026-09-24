"""JSON Lines: jeden záznam pro každý zpracovaný snímek, bez síťového přenosu."""
import json
from dataclasses import asdict


def detection_record(observation, *, sequence, received_at, source):
    measurement = observation.measurement if observation.measured else None
    width, height = observation.frame_size
    valid = observation.confirmed and measurement is not None
    return {
        'schema_version': 1,
        'sequence': sequence,
        'received_at_unix_s': received_at,
        'timestamp_basis': 'host_after_read',
        'source': source,
        'frame_size_px': [width, height],
        'confirmed': observation.confirmed,
        'measured': measurement is not None,
        'valid_pixel_position': valid,
        'measurement_px': asdict(measurement) if measurement is not None else None,
        'tracked_position_px': asdict(observation.target) if observation.target is not None else None,
        'position_kind': ('filtered_measurement' if observation.measured else 'prediction')
                         if observation.target is not None else None,
        'offset_px': {'x': measurement.x-width/2, 'y': measurement.y-height/2} if valid else None,
        'candidate_count': len(observation.circles),
        'processing_ms': observation.processing_ms,
        'geolocation': None,
        'geolocation_status': 'telemetry_and_calibration_not_connected',
    }


class JSONPublisher:
    def __init__(self, stream):
        self.stream = stream

    def publish(self, record):
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()
