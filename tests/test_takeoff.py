import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.takeoff import TakeoffController
from src.approach_simulation import simulate
from src.sitl_takeoff import automatic_takeoff


class TakeoffTests(unittest.TestCase):
    def test_ground_takeoff_then_approach(self):
        rows = simulate(seconds=40, takeoff_height=5)
        waiting = [r for r in rows if not r['takeoff_complete']]
        self.assertGreater(len(waiting), 20)
        self.assertTrue(all(r['north_m'] == r['east_m'] == 0 for r in waiting))
        first = next(r for r in rows if r['takeoff_complete'])
        self.assertLess(abs(first['height_m']-5), .2)
        self.assertLess(abs(first['velocity_up_m_s']), .15)
        self.assertEqual(rows[-1]['state'], 'ARRIVED')

    def test_requires_continuous_settling(self):
        c = TakeoffController(5)
        for i in range(8):
            self.assertFalse(c.update(5, 0, i*.1, i*.1).ready)
        self.assertFalse(c.update(4, 0, .8, .8).ready)
        for i in range(9, 20):
            result = c.update(5, 0, i*.1, i*.1)
        self.assertTrue(result.ready)

    def test_failures_latch(self):
        for height, stamp, now, reason in ((float('nan'), 0, 0, 'TAKEOFF_INVALID_DATA'),
                                          (0, 0, 1, 'TAKEOFF_STALE')):
            c = TakeoffController()
            self.assertEqual(c.update(height, 0, stamp, now).state, reason)
            self.assertFalse(c.update(5, 0, 2, 2).ready)
        c = TakeoffController()
        c.update(5, 0, 0, 0)
        self.assertEqual(c.update(5, 0, 1, 1).state, 'TAKEOFF_STREAM_GAP')

    def test_timeout_and_invalid_height(self):
        c = TakeoffController()
        for i in range(612):
            result = c.update(0, 0, i*.1, i*.1)
        self.assertEqual(result.state, 'TAKEOFF_TIMEOUT')
        for height in (0, -1, 21, float('nan')):
            with self.assertRaises(ValueError):
                TakeoffController(height)

    def test_sitl_requires_confirmation(self):
        with self.assertRaises(ValueError):
            automatic_takeoff(MagicMock(), MagicMock(), 5)

    def test_sitl_sequence_and_rejection(self):
        constants = SimpleNamespace(MAV_MODE_FLAG_SAFETY_ARMED=128,
            MAV_MODE_FLAG_CUSTOM_MODE_ENABLED=1, MAV_CMD_DO_SET_MODE=176,
            MAV_CMD_COMPONENT_ARM_DISARM=400, MAV_CMD_NAV_TAKEOFF=22,
            MAV_RESULT_ACCEPTED=0, MAV_RESULT_IN_PROGRESS=5)
        for scenario in ('success', 'rejected', 'no_climb'):
            rejected = scenario == 'rejected'
            conn = MagicMock(target_system=1, target_component=1)
            conn.mode_mapping.return_value = {'GUIDED': 4}
            current = dict(time=0., mode=0, armed=0, height=0., tick=0)
            pending = []

            def message(kind, **values):
                return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                                       get_srcComponent=lambda: 1, **values)

            def send(*args):
                command = args[2]
                denial = rejected and command == 400
                pending.append(message('COMMAND_ACK', command=command, result=2 if denial else 0))
                if not denial:
                    if command == 176: current['mode'] = 4
                    if command == 400: current['armed'] = 128
                    if command == 22 and scenario != 'no_climb': current['height'] = 5.

            def receive(**kwargs):
                current['time'] += .05
                current['tick'] += 1
                if pending:
                    return pending.pop(0)
                if current['tick'] % 2:
                    return message('HEARTBEAT', custom_mode=current['mode'], base_mode=current['armed'])
                return message('GLOBAL_POSITION_INT', relative_alt=current['height']*1000, vz=0)

            conn.mav.command_long_send.side_effect = send
            conn.recv_match.side_effect = receive
            if rejected:
                with self.assertRaisesRegex(RuntimeError, 'odmítl'):
                    automatic_takeoff(conn, constants, 5, simulator_confirmed=True,
                                      clock=lambda: current['time'], report=lambda _: None)
            elif scenario == 'no_climb':
                with self.assertRaisesRegex(RuntimeError, 'TAKEOFF_TIMEOUT'):
                    automatic_takeoff(conn, constants, 5, simulator_confirmed=True,
                                      clock=lambda: current['time'], report=lambda _: None)
            else:
                automatic_takeoff(conn, constants, 5, simulator_confirmed=True,
                                  clock=lambda: current['time'], report=lambda _: None)
                self.assertGreater(current['time'], 1)
            commands = [call.args[2] for call in conn.mav.command_long_send.call_args_list]
            self.assertEqual(commands, [176, 400] if rejected else [176, 400, 22])
            # Běžné armování bez force parametru.
            self.assertEqual(conn.mav.command_long_send.call_args_list[1].args[4:6], (1, 0))
