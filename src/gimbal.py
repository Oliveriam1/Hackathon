"""Dvouosý závěs kamery (2x MG996R) přes pigpio, stejné hodnoty jako red_tracker.py.

x = náklon kamery doprava (+) / doleva (-), y = dopředu (+) / dozadu (-),
0/0 = kolmo dolů. Na Pi: sudo apt install pigpio python3-pigpio
a sudo systemctl enable --now pigpiod.
"""

from .config import (X_PIN, Y_PIN, X_CENTER_US, Y_CENTER_US, US_PER_DEG,
                     X_DIR, Y_DIR, X_LIMITS, Y_LIMITS)


class _Lgpio:
    """Náhrada pigpio přes lgpio (výchozí v Raspberry Pi OS Bookworm, i Pi 5)."""

    def __init__(self, lgpio):
        self.lgpio = lgpio
        for chip in (0, 4):  # Pi 5 se starším jádrem má piny na gpiochip4
            try:
                self.handle = lgpio.gpiochip_open(chip)
                break
            except lgpio.error:
                continue
        else:
            raise RuntimeError('lgpio: nelze otevřít gpiochip0 ani gpiochip4')
        for pin in (X_PIN, Y_PIN):
            lgpio.gpio_claim_output(self.handle, pin)

    def set_servo_pulsewidth(self, pin, width):
        self.lgpio.tx_servo(self.handle, pin, width)  # 50 Hz; 0 = bez signálu

    def stop(self):
        self.lgpio.gpiochip_close(self.handle)


def _connect():
    """pigpio (hardwarově časované pulzy, jako red_tracker.py), jinak lgpio."""
    import sys
    problems = []
    try:
        import pigpio
        pi = pigpio.pi()
        if pi.connected:
            return pi
        problems.append('pigpio: démon pigpiod neběží (sudo systemctl enable --now pigpiod)')
    except ImportError as error:
        problems.append(f'pigpio: {error}')
    try:
        import lgpio
        return _Lgpio(lgpio)
    except ImportError as error:
        problems.append(f'lgpio: {error}')
    except Exception as error:
        problems.append(f'lgpio: {error}')
    raise RuntimeError(
        f'Serva nelze ovládat ({sys.executable}). ' + '; '.join(problems) +
        '. Instalace: sudo apt install python3-lgpio (nebo pigpio python3-pigpio); '
        've venv použijte python3 -m venv --system-site-packages.')


class Gimbal:
    """Úhly x, y jsou fyzické (stupně). Při otevření najede do 0/0."""

    def __init__(self, pi=None, *, x_dir=X_DIR, y_dir=Y_DIR):
        self.x_dir, self.y_dir = x_dir, y_dir
        self.x = self.y = 0.0
        self.pi = pi

    def open(self):
        if self.pi is None:
            self.pi = _connect()
        self.move_to(0.0, 0.0)

    def pulses(self):
        px = X_CENTER_US + self.x_dir * self.x * US_PER_DEG
        py = Y_CENTER_US + self.y_dir * self.y * US_PER_DEG
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
        try:
            self.open()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()
