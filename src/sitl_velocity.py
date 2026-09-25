"""Testovatelný překlad rychlosti do MAVLink pro SITL. Sám neotevírá spojení.

Pouze pro simulovaný Copter v GUIDED. Neobsahuje armování, vzlet ani změnu režimu.
Connection musí dodat samostatný SITL test; produkční pipeline ho neimportuje.
"""
import math


class SitlVelocityAdapter:
    def __init__(self, connection, *, simulator_confirmed=False):
        if not simulator_confirmed:
            raise ValueError('Adaptér je pouze pro explicitně potvrzený SITL test.')
        self.connection = connection

    def send(self, command):
        n, e = command.north_m_s, command.east_m_s
        if not math.isfinite(n) or not math.isfinite(e) or math.hypot(n, e) > 2.000001:
            raise ValueError('Neplatný nebo příliš rychlý simulační povel.')
        # MAV_FRAME_LOCAL_NED=1, velocity-only type_mask=3527, vz=0 (držení výšky).
        self.connection.mav.set_position_target_local_ned_send(
            0, self.connection.target_system, self.connection.target_component,
            1, 3527, 0, 0, 0, n, e, 0, 0, 0, 0, 0, 0)

    def hold(self):
        from .approach import ApproachCommand
        self.send(ApproachCommand('HOLD'))
