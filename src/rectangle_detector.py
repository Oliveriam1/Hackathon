"""Konvexní čtyřúhelníky včetně čtverců a perspektivně zkosených terčů."""
import cv2
import numpy as np

MIN_AREA = 200
MIN_SIDE = 8
_CLOSE_KERNEL = np.ones((3, 3), np.uint8)


def _quadrilateral(contour):
    # Zaoblené nebo rozmazané rohy potřebují větší toleranci aproximace.
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.03, 0.045):
        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(polygon) < 4:
            return None
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            break
    else:
        return None
    # Obrys musí polygon těsně sledovat. Zubatá skvrna textury (tráva)
    # má mnohem delší obvod a jinou plochu než její aproximace.
    polygon_perimeter = cv2.arcLength(polygon, True)
    if perimeter > 1.25 * polygon_perimeter:
        return None
    # U malých terčů mění posun hrany o 1-2 px plochu o desítky procent.
    polygon_area = cv2.contourArea(polygon)
    if abs(cv2.contourArea(contour) - polygon_area) > 0.1*polygon_area + 1.5*polygon_perimeter:
        return None
    return polygon


def find_rectangles(edges):
    # Uzavření zacelí jednopixelové mezery v hranách reálné kamery, ale slije
    # blízké rovnoběžné hrany (tenký rám). Proto obrysy z obou variant.
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    contours = [contour for source in (edges, closed)
                for contour in cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]]
    height, width = edges.shape
    rectangles = []
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        polygon = _quadrilateral(contour)
        if polygon is None:
            continue
        area = cv2.contourArea(polygon)
        if area < MIN_AREA:
            continue
        points = polygon[:, 0, :].astype(float)
        if (points[:, 0].min() <= 1 or points[:, 1].min() <= 1
                or points[:, 0].max() >= width-2 or points[:, 1].max() >= height-2):
            continue
        sides = np.roll(points, -1, axis=0) - points
        if np.linalg.norm(sides, axis=1).min() < MIN_SIDE:
            continue
        center = points.mean(axis=0)
        if any(np.linalg.norm(center - old[:, 0, :].mean(axis=0)) < 8
               and 0.8 < area / cv2.contourArea(old) < 1.25 for old in rectangles):
            continue
        rectangles.append(polygon)
    return rectangles


def enclosing_rectangle(circle, rectangles, margin=2):
    """Nejmenší čtyřúhelník, ve kterém leží celý obvod kolečka s rezervou."""
    boundary = circle.boundary()
    matches = [rect for rect in rectangles
               if min(cv2.pointPolygonTest(rect, (float(x), float(y)), True) for x, y in boundary) > margin]
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
