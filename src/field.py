"""Optional perspective grid for debugging; never a source of GNSS coordinates."""

import cv2
import numpy as np

from .models import FieldCorners, FieldPoint, GridCell, Point


def _column_label(column: int) -> str:
    """Convert a zero-based column to A..Z, AA..AZ, BA, etc."""
    label = ""
    number = column + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform finite points, rejecting points on the projective horizon."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Image coordinates must be finite")
    terms = points[:, 0, :] * matrix[2, :2]
    denominators = terms.sum(axis=1) + matrix[2, 2]
    scale = np.abs(terms).sum(axis=1) + abs(matrix[2, 2])
    if np.any(np.abs(denominators) <= 1e-12 * scale):
        raise ValueError("Point cannot be projected: homography denominator is zero")
    transformed = cv2.perspectiveTransform(points, matrix)
    if not np.isfinite(transformed).all():
        raise ValueError("Homography produced non-finite coordinates")
    return transformed[:, 0, :]


class FieldGrid:
    """Map image pixels onto a normalized field and a configurable grid.

    Field axes start at the top-left corner and increase right/down. The
    virtual plane has arbitrary units, not physical or geographic distances.
    """

    def __init__(
        self,
        rows: int = 10,
        columns: int = 10,
        virtual_width: int = 1000,
        virtual_height: int = 1000,
    ) -> None:
        for name, value in (
            ("rows", rows),
            ("columns", columns),
            ("virtual_width", virtual_width),
            ("virtual_height", virtual_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.rows = rows
        self.columns = columns
        self.virtual_width = virtual_width
        self.virtual_height = virtual_height
        self._homography: np.ndarray | None = None

    def set_corners(self, corners: FieldCorners) -> None:
        """Calibrate from four distinct, ordered corners of a convex field.

        Reject degenerate, concave or crossed quadrilaterals. Failed
        recalibration leaves the previous valid homography intact.
        """
        source = np.array(
            [(point.x, point.y) for point in (
                corners.top_left,
                corners.top_right,
                corners.bottom_right,
                corners.bottom_left,
            )],
            dtype=np.float32,
        )
        if not np.isfinite(source).all() or not cv2.isContourConvex(source):
            raise ValueError("Field corners must form a finite, convex quadrilateral")
        destination = np.array(
            [(0, 0), (self.virtual_width, 0),
             (self.virtual_width, self.virtual_height), (0, self.virtual_height)],
            dtype=np.float32,
        )
        try:
            matrix = cv2.getPerspectiveTransform(source, destination)
            inverse = np.linalg.inv(matrix)
        except (cv2.error, np.linalg.LinAlgError) as exc:
            raise ValueError("Field corners produce an invalid homography") from exc
        if not np.isfinite(matrix).all() or not np.isfinite(inverse).all():
            raise ValueError("Field corners produce an invalid homography")
        projected = _transform_points(source, matrix)
        if not np.allclose(projected, destination, rtol=1e-6, atol=1e-5):
            raise ValueError("Field homography does not map the supplied corners")
        self._homography = matrix

    def pixel_to_field(self, point: Point) -> FieldPoint:
        """Project a pixel and clamp its normalized field position to [0, 1].

        Outside points map to the field boundary. Calibration must describe
        the current camera view; recalibrate when that view changes.
        """
        if self._homography is None:
            raise RuntimeError("Field homography has not been initialized")
        virtual = _transform_points(np.array([[point.x, point.y]]), self._homography)[0]
        normalized = np.clip(
            virtual / (self.virtual_width, self.virtual_height), 0.0, 1.0
        )
        return FieldPoint(x=float(normalized[0]), y=float(normalized[1]))

    def field_to_grid(self, point: FieldPoint) -> GridCell:
        """Map normalized coordinates to zero-based indices and a cell label.

        Internal boundaries belong to the next cell right/down. Exactly 1.0
        belongs to the last column/row, so (1, 1) is J10 in a 10 by 10 grid.
        """
        column = min(int(point.x * self.columns), self.columns - 1)
        row = min(int(point.y * self.rows), self.rows - 1)
        return GridCell(row=row, column=column, label=f"{_column_label(column)}{row + 1}")

    def pixel_to_grid(self, point: Point) -> tuple[FieldPoint, GridCell]:
        """Return both the normalized field position and its grid cell."""
        field_point = self.pixel_to_field(point)
        return field_point, self.field_to_grid(field_point)


def draw_grid_overlay(
    frame: np.ndarray,
    corners: FieldCorners,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Draw a perspective grid and sparse cell labels on a copy of a BGR frame."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.size == 0
    ):
        raise ValueError("frame must be a non-empty uint8 BGR image (H, W, 3)")
    grid = FieldGrid(rows=rows, columns=columns)
    grid.set_corners(corners)
    assert grid._homography is not None
    inverse = np.linalg.inv(grid._homography)
    width, height = grid.virtual_width, grid.virtual_height
    overlay = frame.copy()

    # Lines are straight in the field plane; project their endpoints to the image.
    lines = [((column * width / columns, 0), (column * width / columns, height))
             for column in range(columns + 1)]
    lines += [((0, row * height / rows), (width, row * height / rows))
              for row in range(rows + 1)]
    projected = _transform_points(np.array(lines), inverse).reshape(-1, 2, 2)
    for index, endpoints in enumerate(projected):
        start, end = np.rint(endpoints).astype(np.int32)
        border = index in (0, columns, columns + 1, columns + 1 + rows)
        cv2.line(overlay, tuple(start), tuple(end), (255, 200, 0),
                 2 if border else 1, cv2.LINE_AA)

    # Label at most five cells, skipping cells too small to hold the text.
    samples = sorted({(0, 0), (0, columns - 1), (rows - 1, 0),
                      (rows - 1, columns - 1), (rows // 2, columns // 2)})
    for row, column in samples:
        left, right = column * width / columns, (column + 1) * width / columns
        top, bottom = row * height / rows, (row + 1) * height / rows
        cell = _transform_points(
            np.array([(left, top), (right, top), (right, bottom), (left, bottom),
                      ((left + right) / 2, (top + bottom) / 2)]), inverse
        )
        label = f"{_column_label(column)}{row + 1}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        clearance = cv2.pointPolygonTest(cell[:4].astype(np.float32), tuple(cell[4]), True)
        if clearance < np.hypot(text_width, text_height + baseline) / 2 + 2:
            continue
        origin = (round(cell[4, 0] - text_width / 2), round(cell[4, 1] + text_height / 2))
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay
