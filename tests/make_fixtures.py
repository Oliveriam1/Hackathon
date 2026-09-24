"""Regenerate the five deterministic synthetic static-image fixtures."""

import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    output = Path(__file__).parent / "assets"
    output.mkdir(exist_ok=True)
    plane = np.full((400, 600, 3), 110, np.uint8)
    cv2.rectangle(plane, (4, 4), (595, 395), (230, 230, 230), 8)
    cv2.circle(plane, (285, 190), 55, (25, 25, 25), 8)
    source = np.array([[0, 0], [599, 0], [599, 399], [0, 399]], np.float32)
    cases = [
        ("front", [[150, 100], [750, 100], [750, 500], [150, 500]]),
        ("perspective", [[170, 100], [760, 135], [710, 520], [120, 470]]),
        ("rotated", [[200, 80], [780, 200], [690, 530], [110, 410]]),
        ("dark", [[150, 100], [750, 100], [750, 500], [150, 500]]),
        ("noise_magenta", [[170, 100], [760, 135], [710, 520], [120, 470]]),
    ]
    expected = []
    for name, destination in cases:
        matrix = cv2.getPerspectiveTransform(source, np.array(destination, np.float32))
        frame = cv2.warpPerspective(plane, matrix, (900, 600), borderValue=(110, 110, 110))
        if name == "dark":
            gradient = np.linspace(0.45, 0.8, 900)[None, :, None]
            frame = np.clip(frame * gradient, 0, 255).astype(np.uint8)
        if name == "noise_magenta":
            noise = np.random.default_rng(42).normal(0, 5, frame.shape)
            frame = np.clip(frame.astype(float) + [35, -20, 55] + noise, 0, 255).astype(np.uint8)
        point = cv2.perspectiveTransform(np.array([[[285, 190]]], np.float32), matrix)[0, 0]
        # imencode + tofile: cv2.imwrite na Windows neumí cestu s diakritikou/azbukou.
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("Could not write fixture")
        encoded.tofile(output / f"{name}.png")
        expected.append({"file": f"{name}.png", "center": [float(point[0]), float(point[1])]})
    (output / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")


if __name__ == "__main__":
    main()
