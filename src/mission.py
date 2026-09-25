"""Hranice mezi pozorováním a autonomní misí.

Fáze 1: mise zůstává neimplementovaná. Vizuální LOCKED není letový stav.
Přechody vzlet/hledání/přelet zde přibudou po implementaci řídicích komponent.
"""


class Mission:
    def snapshot(self):
        return {'enabled': False, 'state': 'NOT_IMPLEMENTED',
                'missing': ['flight_adapter', 'zone_boundary', 'gimbal_feedback',
                            'exposure_telemetry_synchronization']}
