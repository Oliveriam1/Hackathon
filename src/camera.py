"""Otevření kamery a čtení snímků přes OpenCV."""

import cv2


class Camera:
    def __init__(self, index: int | str = 0):
        """Index kamery, nebo cesta k videozáznamu."""
        self.index = index
        self._capture = None

    def open(self):
        if self._capture is not None:
            return
        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            if isinstance(self.index, str):
                raise RuntimeError(f"Video {self.index} nelze otevřít.")
            raise RuntimeError(
                f"Kameru s indexem {self.index} nelze otevřít. "
                "Zkontrolujte připojení, oprávnění ke kameře a zda ji nepoužívá jiný program. "
                "Případně zkuste --camera 1."
            )
        self._capture = capture

    def read(self):
        if self._capture is None:
            raise RuntimeError("Kamera není otevřená.")
        success, frame = self._capture.read()
        if not success or frame is None or frame.size == 0:
            raise RuntimeError("Nepodařilo se načíst snímek. Zkontrolujte připojení kamery.")
        return frame

    def rewind(self):
        """Videozáznam znovu od začátku."""
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
