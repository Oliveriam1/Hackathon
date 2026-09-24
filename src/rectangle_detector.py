"""Konvexní čtyřúhelníky blízké obdélníku v obraze."""
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
        if lengths.min() < 25 or lengths.max() / lengths.min() > 5:
            continue
        cosines = np.sum(sides * np.roll(sides, 1, axis=0), axis=1) / (lengths * np.roll(lengths, 1))
        if np.max(np.abs(cosines)) > 0.4:
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
