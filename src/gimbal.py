"""Dvouosý závěs kamery (2x MG996R) přes pigpio, stejné hodnoty jako red_tracker.py.

x = náklon kamery doprava (+) / doleva (-), y = dopředu (+) / dozadu (-),
0/0 = kolmo dolů. Na Pi: sudo apt install pigpio python3-pigpio
a sudo systemctl enable --now pigpiod.
"""

X_PIN, Y_PIN = 18, 13                    # BCM
X_CENTER_US, Y_CENTER_US = 1500, 1500    # pulz, při kterém kamera míří kolmo dolů
US_PER_DEG = 1000.0 / 90.0               # 500..2500 us na 180°
X_DIR, Y_DIR = 1, 1                      # -1, pokud se osa točí opačně
X_LIMITS = (-60.0, 60.0)
Y_LIMITS = (-45.0, 45.0)


class Gimbal:
    """Úhly x, y jsou fyzické (stupně). Při otevření najede do 0/0."""

    def __init__(self, pi=None):
        self.x = self.y = 0.0
        self.pi = pi

    def open(self):
        if self.pi is None:
            try:
                import pigpio
            except ImportError as error:
                raise RuntimeError('pigpio není dostupné: sudo apt install pigpio python3-pigpio') from error
            self.pi = pigpio.pi()
            if not self.pi.connected:
                self.pi = None
                raise RuntimeError('pigpiod neběží: sudo systemctl start pigpiod')
        self.move_to(0.0, 0.0)

    def pulses(self):
        px = X_CENTER_US + X_DIR * self.x * US_PER_DEG
        py = Y_CENTER_US + Y_DIR * self.y * US_PER_DEG
        return int(max(500, min(2500, px))), int(max(500, min(2500, py)))

    def move_to(self, x, y):
        self.x = max(X_LIMITS[0], min(X_LIMITS[1], x))
        self.y = max(Y_LIMITS[0], min(Y_LIMITS[1], y))
        if self.pi is not None:
            px, py = self.pulses()
            self.pi.set_servo_pulsewidth(X_PIN, px)
            self.pi.set_servo_pulsewidth(Y_PIN, py)

    def move_by(self, dx, dy):
        self.move_to(self.x + dx, self.y + dy)

    def close(self):
        # Pulz 0 = servo bez signálu (jako red_tracker.py při ukončení).
        if self.pi is not None:
            pi, self.pi = self.pi, None
            try:
                pi.set_servo_pulsewidth(X_PIN, 0)
                pi.set_servo_pulsewidth(Y_PIN, 0)
            finally:
                pi.stop()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
