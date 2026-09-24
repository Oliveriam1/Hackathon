"""Field mapping and overlay tests using synthetic image geometry."""

import unittest

import cv2
import numpy as np

from src.detector import ColorDetector
from src.field import FieldGrid, draw_grid_overlay
from src.models import FieldCorners, FieldPoint, GridCell, Point


SQUARE = FieldCorners(Point(0, 0), Point(1000, 0), Point(1000, 1000), Point(0, 1000))
TRAPEZOID = FieldCorners(Point(100, 100), Point(300, 100), Point(400, 400), Point(0, 400))


class FieldGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = FieldGrid()
        self.grid.set_corners(SQUARE)

    def test_square_center_is_f6(self) -> None:
        point, cell = self.grid.pixel_to_grid(Point(500, 500))
        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 0.5)
        self.assertEqual(cell, GridCell(row=5, column=5, label="F6"))

    def test_top_left_is_a1(self) -> None:
        point, cell = self.grid.pixel_to_grid(Point(0, 0))
        self.assertEqual(point, FieldPoint(0.0, 0.0))
        self.assertEqual(cell.label, "A1")

    def test_near_bottom_right_is_j10(self) -> None:
        point, cell = self.grid.pixel_to_grid(Point(999, 999))
        self.assertAlmostEqual(point.x, 0.999)
        self.assertAlmostEqual(point.y, 0.999)
        self.assertEqual(cell.label, "J10")

    def test_all_exact_corners(self) -> None:
        for pixel, expected, label in (
            (SQUARE.top_left, FieldPoint(0, 0), "A1"),
            (SQUARE.top_right, FieldPoint(1, 0), "J1"),
            (SQUARE.bottom_right, FieldPoint(1, 1), "J10"),
            (SQUARE.bottom_left, FieldPoint(0, 1), "A10"),
        ):
            with self.subTest(label=label):
                point, cell = self.grid.pixel_to_grid(pixel)
                self.assertEqual(point, expected)
                self.assertEqual(cell.label, label)

    def test_perspective_diagonal_intersection_maps_to_center(self) -> None:
        self.grid.set_corners(TRAPEZOID)
        # The diagonals intersect at (200, 200), not the bounding-box center.
        point = self.grid.pixel_to_field(Point(200, 200))
        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 0.5)
        for pixel, expected in (
            (TRAPEZOID.top_left, FieldPoint(0, 0)),
            (TRAPEZOID.top_right, FieldPoint(1, 0)),
            (TRAPEZOID.bottom_right, FieldPoint(1, 1)),
            (TRAPEZOID.bottom_left, FieldPoint(0, 1)),
        ):
            with self.subTest(pixel=pixel):
                actual = self.grid.pixel_to_field(pixel)
                self.assertAlmostEqual(actual.x, expected.x)
                self.assertAlmostEqual(actual.y, expected.y)

    def test_uncalibrated_mapping_raises(self) -> None:
        grid = FieldGrid()
        for method in (grid.pixel_to_field, grid.pixel_to_grid):
            with self.assertRaisesRegex(RuntimeError, "has not been initialized"):
                method(Point(500, 500))
        self.assertEqual(grid.field_to_grid(FieldPoint(0.642, 0.317)).label, "G4")

    def test_outside_pixels_are_clamped(self) -> None:
        self.assertEqual(self.grid.pixel_to_field(Point(-100, 1200)), FieldPoint(0, 1))
        self.assertEqual(self.grid.pixel_to_field(Point(1200, -100)), FieldPoint(1, 0))

    def test_custom_dimensions(self) -> None:
        grid = FieldGrid(rows=4, columns=8, virtual_width=2000, virtual_height=500)
        grid.set_corners(SQUARE)
        point, cell = grid.pixel_to_grid(Point(500, 500))
        self.assertEqual(point, FieldPoint(0.5, 0.5))
        self.assertEqual(cell, GridCell(row=2, column=4, label="E3"))

    def test_labels_beyond_z_and_single_cell(self) -> None:
        grid = FieldGrid(rows=1, columns=28)
        self.assertEqual(grid.field_to_grid(FieldPoint(26 / 28, 0)).label, "AA1")
        self.assertEqual(grid.field_to_grid(FieldPoint(1, 1)).label, "AB1")
        self.assertEqual(FieldGrid(1, 1).field_to_grid(FieldPoint(1, 1)).label, "A1")

    def test_grid_boundaries(self) -> None:
        self.assertEqual(self.grid.field_to_grid(FieldPoint(0.099999, 0.099999)).label, "A1")
        self.assertEqual(self.grid.field_to_grid(FieldPoint(0.1, 0.1)).label, "B2")

    def test_invalid_dimensions(self) -> None:
        for name in ("rows", "columns", "virtual_width", "virtual_height"):
            for value in (0, -1, 2.5, float("nan"), float("inf"), True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        FieldGrid(**{name: value})

    def test_invalid_field_points(self) -> None:
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FieldPoint(value, 0)
                with self.assertRaises(ValueError):
                    FieldPoint(0, value)

    def test_degenerate_or_misordered_corners(self) -> None:
        for points in (
            [(0, 0), (1, 1), (2, 2), (3, 3)],  # Collinear.
            [(0, 0), (100, 0), (100, 0), (0, 100)],  # Repeated corner.
            [(0, 0), (50, 0), (100, 0), (0, 100)],  # Three collinear corners.
            [(0, 0), (100, 100), (100, 0), (0, 100)],  # Crossing edges.
            [(0, 0), (100, 0), (20, 20), (0, 100)],  # Concave.
            [(0, 0), (float("nan"), 0), (100, 100), (0, 100)],
        ):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    self.grid.set_corners(FieldCorners(*(Point(x, y) for x, y in points)))
        # A rejected update must not corrupt the preceding valid calibration.
        self.assertEqual(self.grid.pixel_to_field(Point(500, 500)), FieldPoint(0.5, 0.5))

    def test_horizon_and_nonfinite_pixels_are_rejected(self) -> None:
        self.grid.set_corners(TRAPEZOID)
        with self.assertRaisesRegex(ValueError, "denominator"):
            self.grid.pixel_to_field(Point(200, -200))
        with self.assertRaises(ValueError):
            self.grid.pixel_to_field(Point(float("nan"), 0))

    def test_recalibration(self) -> None:
        self.grid.set_corners(TRAPEZOID)
        self.assertAlmostEqual(self.grid.pixel_to_field(Point(200, 200)).x, 0.5)
        self.grid.set_corners(SQUARE)
        self.assertAlmostEqual(self.grid.pixel_to_field(Point(200, 200)).x, 0.2)

    def test_multiple_detections_use_the_same_grid(self) -> None:
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.rectangle(frame, (130, 130), (170, 170), (0, 0, 255), -1)
        cv2.rectangle(frame, (630, 330), (670, 370), (0, 0, 255), -1)
        cv2.rectangle(frame, (480, 480), (520, 520), (0, 255, 0), -1)
        detector = ColorDetector()
        detections = detector.detect_red(frame)
        self.assertEqual(len(detections), 2)
        self.assertEqual(
            {self.grid.pixel_to_grid(detection.center)[1].label for detection in detections},
            {"B2", "G4"},
        )
        green = detector.detect_green(frame)
        self.assertIsNotNone(green)
        self.assertTrue(detector.get_centering(frame, green).centered)


class GridOverlayTests(unittest.TestCase):
    def test_overlay_copies_frame_and_respects_perspective(self) -> None:
        frame = np.zeros((500, 500, 3), dtype=np.uint8)
        overlay = draw_grid_overlay(frame, TRAPEZOID, rows=2, columns=2)
        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual(overlay.dtype, frame.dtype)
        self.assertFalse(np.shares_memory(overlay, frame))
        self.assertFalse(frame.any())
        # Perspective midpoint is y=200, not the linear midpoint y=250.
        self.assertTrue(overlay[200, 150].any())
        self.assertFalse(overlay[250, 150].any())
        self.assertTrue(overlay[100, 200].any())
        self.assertTrue(overlay[400, 200].any())
        self.assertFalse(overlay[20, 20].any())

    def test_overlay_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            draw_grid_overlay(np.zeros((10, 10), np.uint8), SQUARE, 10, 10)
        with self.assertRaises(ValueError):
            draw_grid_overlay(np.zeros((10, 10, 3), np.uint8), SQUARE, 0, 10)


if __name__ == "__main__":
    unittest.main()
