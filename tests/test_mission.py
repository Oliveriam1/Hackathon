import math
import unittest
from dataclasses import replace
from src.config import StartReference
from src.field import GroundPoint, FieldMap
from src.approach import VehicleState
from src.mission_controller import MissionController, MissionSettings, MissionInput, CommandFeedback
from src.mission_simulation import simulate_mission
from src.mission_inputs import target_from_drone_data
from src.geolocation import offset_to_latlon


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.settings = MissionSettings(StartReference(50, 14), (-10, 10, -10, 10))

    def input(self, now, **kwargs):
        corners = tuple(GroundPoint(*p) for p in ((-10, -10), (10, -10), (10, 10), (-10, 10)))
        return replace(MissionInput(VehicleState(GroundPoint(0, 0), 0, 0, now),
                       FieldMap(corners, now), 0., 0., now, camera_ready=True), **kwargs)

    def test_full_pipeline_and_result(self):
        rows = simulate_mission(self.settings)
        states = {r['command']['state'] for r in rows}
        for state in ('SETTING_GUIDED', 'ARMING', 'TAKING_OFF', 'SEARCHING', 'APPROACHING', 'HOLDING_TARGET'):
            self.assertIn(state, states)
        self.assertEqual(rows[-1]['command']['state'], 'HOLDING_TARGET')
        self.assertLessEqual(rows[-1]['distance_m'], .4)
        self.assertIsNotNone(rows[-1]['mission']['target_result'])
        requests = [r['command']['action'] for r in rows if r['command']['request_id'] is not None]
        self.assertEqual(requests, ['SET_GUIDED', 'ARM', 'TAKEOFF'])
        self.assertTrue(all(abs(r['north_m']) < 9.5 and abs(r['east_m']) < 9.5 for r in rows))
        for row in rows:
            cmd = row['command']
            self.assertLessEqual(math.hypot(cmd['north_m_s'], cmd['east_m_s']), 2.00001)
            if not row['mission']['takeoff_complete']:
                self.assertEqual((cmd['north_m_s'], cmd['east_m_s']), (0, 0))

    def test_no_target_and_camera_lock_required(self):
        rows = simulate_mission(self.settings, no_target=True)
        self.assertEqual(rows[-1]['command']['state'], 'SEARCH_COMPLETE')
        self.assertIsNone(rows[-1]['mission']['target_result'])
        rows = simulate_mission(self.settings, fault='camera_unlocked', fault_at=0)
        self.assertEqual(rows[-1]['command']['state'], 'CENTERING_CAMERA')
        self.assertIsNone(rows[-1]['mission']['target_result'])

    def test_faults_and_manual_are_terminal(self):
        for fault, expected in (('telemetry', 'FAILSAFE'), ('map', 'FAILSAFE'),
                                ('camera', 'FAILSAFE'), ('manual', 'MANUAL'), ('stop', 'ABORTED')):
            rows = simulate_mission(self.settings, seconds=25, fault=fault, fault_at=20)
            affected = [r for r in rows if r['time_s'] >= 20]
            self.assertTrue(all(r['command']['state'] == expected for r in affected), fault)
            self.assertTrue(all(r['command']['north_m_s'] == r['command']['east_m_s'] == 0 for r in affected))
            if fault == 'manual':
                self.assertTrue(all(r['command']['action'] == 'RELEASE' for r in affected))

    def test_rejected_arm_and_takeoff_timeout(self):
        rows = simulate_mission(self.settings, seconds=1, fault='reject_arm')
        self.assertEqual(rows[-1]['command']['reason'], 'COMMAND_REJECTED')
        self.assertFalse(any(r['command']['action'] == 'TAKEOFF' for r in rows))
        rows = simulate_mission(self.settings, seconds=65, fault='no_climb')
        self.assertEqual(rows[-1]['command']['reason'], 'TAKEOFF_TIMEOUT')

    def test_unknown_map_blocks_preflight(self):
        mission = MissionController(self.settings)
        command = mission.update(self.input(0, field=FieldMap()), 0)
        self.assertEqual(command.state, 'PREFLIGHT')
        self.assertIsNone(command.request_id)
        self.assertEqual(command.reason, 'ZONE_UNKNOWN')

    def test_matching_ack_and_state_both_required(self):
        mission = MissionController(self.settings)
        request = mission.update(self.input(0), 0)
        cmd = mission.update(self.input(.1, guided=True, feedback=CommandFeedback(999, True)), .1)
        self.assertEqual(cmd.state, 'SETTING_GUIDED')
        cmd = mission.update(self.input(.2, feedback=CommandFeedback(request.request_id, True)), .2)
        self.assertEqual(cmd.state, 'SETTING_GUIDED')
        cmd = mission.update(self.input(.3, guided=True), .3)
        self.assertEqual(cmd.action, 'ARM')

    def test_ack_timeout_clock_gap_and_no_auto_resume(self):
        mission = MissionController(self.settings)
        for i in range(102):
            cmd = mission.update(self.input(i*.1), i*.1)
        self.assertEqual(cmd.reason, 'COMMAND_ACK_TIMEOUT')
        mission = MissionController(self.settings)
        mission.update(self.input(0), 0)
        self.assertEqual(mission.update(self.input(1), 1).state, 'FAILSAFE')
        self.assertEqual(mission.update(self.input(1.1), 1.1).state, 'FAILSAFE')

    def test_loss_after_arrival_then_reacquire(self):
        base = simulate_mission(self.settings)
        arrived = next(r['time_s'] for r in base if r['command']['state'] == 'HOLDING_TARGET')
        rows = simulate_mission(self.settings, target_loss=(arrived+1, arrived+4))
        self.assertTrue(any(r['command']['state'] == 'TARGET_LOST_HOLD' for r in rows))
        self.assertEqual(rows[-1]['command']['state'], 'HOLDING_TARGET')

    def test_visual_contract_conversion_and_stale_rejection(self):
        lat, lon = offset_to_latlon(50, 14, 8, 5)
        data = dict(live_control_input_valid=True, measurement_valid=True, input_source='camera',
                    target_id=7, world_position_status='ESTIMATED',
                    world_position=dict(latitude_deg=lat, longitude_deg=lon,
                                        horizontal_error_estimate_m=.2, datum='WGS84'))
        target = target_from_drone_data(data, self.settings.start, sampled_at=1, now=1.1)
        self.assertAlmostEqual(target.position.north_m, 8, places=5)
        self.assertAlmostEqual(target.position.east_m, 5, places=5)
        self.assertIsNone(target_from_drone_data(data, self.settings.start, sampled_at=1, now=2))
        self.assertIsNone(target_from_drone_data({**data, 'input_source': 'image'},
                                               self.settings.start, sampled_at=1, now=1.1))

    def test_moving_target_and_mission_deadline(self):
        rows = simulate_mission(self.settings, moving=True)
        self.assertNotEqual(rows[-1]['command']['state'], 'FAILSAFE')
        self.assertLess(rows[-1]['distance_m'], 1.)
        rows = simulate_mission(replace(self.settings, timeout_s=10), seconds=12)
        self.assertEqual(rows[-1]['command']['reason'], 'MISSION_TIMEOUT')
