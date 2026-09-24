"""Ruční označení pixelu; nejde o sledování pohybujícího se objektu."""

import cv2


class ManualTarget:
    def __init__(self):
        self.point = None
        self.frame_size = None

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.point = None
        elif event == cv2.EVENT_LBUTTONDOWN and self.frame_size is not None:
            width, height = self.frame_size
            if 0 <= x < width and 0 <= y < height:
                self.point = (x, y)
                print(f"Cíl: u={x} px, v={y} px", flush=True)

    def annotate(self, frame):
        height, width = frame.shape[:2]
        if self.frame_size != (width, height):
            self.point = None
        self.frame_size = (width, height)
        image = frame.copy()
        cv2.drawMarker(image, (width // 2, height // 2), (255, 255, 255),
                       cv2.MARKER_CROSS, 16, 1)
        if self.point is not None:
            u, v = self.point
            cv2.drawMarker(image, self.point, (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(image, f"u={u} v={v} px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return image
