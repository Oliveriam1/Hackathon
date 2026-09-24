"""Small JSON configuration helpers; no hardware or numerical imports."""

import json
from pathlib import Path

from .models import CameraIntrinsics, CameraMount

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_camera_calibration(path: Path) -> tuple[CameraIntrinsics, CameraMount]:
    try:
        data = read_json(path)
        return CameraIntrinsics(**data["intrinsics"]), CameraMount(**data["mount"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"CAMERA CALIBRATION REQUIRED: {path}: {exc}") from exc
