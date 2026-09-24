"""Konvexní čtyřúhelníky včetně čtverců a perspektivně zkosených terčů."""
import cv2
import numpy as np


def find_rectangles(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = edges.shape
    rectangles = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = cv2.contourArea(polygon)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon) or area < 1200:
            continue
        points = polygon[:, 0, :].astype(float)
        if (points[:, 0].min() <= 1 or points[:, 1].min() <= 1
                or points[:, 0].max() >= width-2 or points[:, 1].max() >= height-2):
            continue
        sides = np.roll(points, -1, axis=0) - points
        lengths = np.linalg.norm(sides, axis=1)
        if lengths.min() < 20:
            continue
        center = points.mean(axis=0)
        if any(np.linalg.norm(center - old[:, 0, :].mean(axis=0)) < 8
               and 0.8 < area / cv2.contourArea(old) < 1.25 for old in rectangles):
            continue
        rectangles.append(polygon)
    return rectangles


def enclosing_rectangle(circle, rectangles):
    matches = [rect for rect in rectangles
               if cv2.pointPolygonTest(rect, (circle.x, circle.y), True) > circle.radius + 2]
    return min(matches, key=cv2.contourArea) if matches else None


def find_faint_quadrilaterals(frame):
    """Záloha pro dlouhé slabé hrany s malými mezerami v rozích."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 1)
    edges = cv2.Canny(gray, 5, 15)
    size = min(gray.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                           minLineLength=max(40, int(size*0.25)),
                           maxLineGap=max(10, int(size*0.1)))
    if lines is None:
        return []
    segments = lines.reshape(-1, 4)
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    canvas = np.zeros_like(gray)
    # Omezený počet segmentů i prodloužení: nespojujeme vzdálené nesouvisející hrany.
    for index in np.argsort(lengths)[-60:]:
        start = segments[index, :2].astype(float)
        end = segments[index, 2:].astype(float)
        direction = (end-start)/lengths[index]*max(5, size*0.025)
        cv2.line(canvas, tuple(np.rint(start-direction).astype(int)),
                 tuple(np.rint(end+direction).astype(int)), 255, 3)
    return find_rectangles(canvas)
