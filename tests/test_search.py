import unittest
from src.approach_simulation import simulate
from src.search_planner import RectangleSweep
from src.search_mission import SearchMission
from src.approach import VehicleState, TargetEstimate
from src.field import GroundPoint


class SearchTests(unittest.TestCase):
    def test_route_in_bounds_and_alternating(self):
        sweep = RectangleSweep((-10, 10, -10, 10))
        self.assertEqual(len(sweep.points), 12)
        for point in sweep.points:
            self.assertTrue(-9 <= point.north_m <= 9 and -9 <= point.east_m <= 9)
        for i in range(2, len(sweep.points), 2):
            self.assertEqual(sweep.points[i-1].north_m, sweep.points[i].north_m)
            self.assertLessEqual(sweep.points[i].east_m-sweep.points[i-1].east_m, 4)

    def test_unknown_or_invalid_field_rejected(self):
        for bounds in ((), (0, 0, 0, 0), (float('nan'), 10, -10, 10)):
            with self.assertRaises(ValueError):
                RectangleSweep(bounds)
        with self.assertRaises(ValueError):
            simulate(search=True, field_bounds=(1, 10, 1, 10))

    def test_takeoff_search_detection_arrival(self):
        rows = simulate(search=True, takeoff_height=5, seconds=300)
        self.assertFalse(rows[0]['target_visible'])
        self.assertTrue(any(r['state'] == 'SEARCHING' for r in rows))
        self.assertEqual(rows[-1]['state'], 'ARRIVED')
        self.assertLessEqual(rows[-1]['distance_m'], .4)
        self.assertTrue(all(abs(r['north_m']) < 10 and abs(r['east_m']) < 10 for r in rows))
        for r in rows:
            if not r['takeoff_complete']:
                self.assertEqual((r['command_north_m_s'], r['command_east_m_s']), (0, 0))

    def test_without_target_completes_and_holds(self):
        rows = simulate(search=True, no_target=True, seconds=300)
        self.assertEqual(rows[-1]['state'], 'SEARCH_COMPLETE')
        self.assertEqual(rows[-1]['search_waypoint'], rows[-1]['search_waypoint_count'])
        self.assertEqual(rows[-1]['command_north_m_s'], 0)
        self.assertEqual(rows[-1]['command_east_m_s'], 0)

    def test_loss_resumes_same_waypoint_and_stale_target_does_not_lock(self):
        mission = SearchMission(RectangleSweep((-10, 10, -10, 10)))
        def step(t, target=None):
            return mission.update(VehicleState(GroundPoint(0, 0), 0, 0, t), target, t)
        step(0, TargetEstimate(GroundPoint(1, 1), 0, 'dot'))
        self.assertEqual(step(.1).state, 'TARGET_LOST_HOLD')
        for i in range(2, 22):
            cmd = step(i*.1)
        self.assertEqual(cmd.state, 'SEARCHING')
        self.assertEqual(mission.index, 0)
        cmd = step(2.2, TargetEstimate(GroundPoint(1, 1), 0, 'dot'))
        self.assertEqual(cmd.state, 'SEARCHING')
        self.assertFalse(mission.tracking)

    def test_stale_vehicle_blocks_search(self):
        mission = SearchMission(RectangleSweep((-10, 10, -10, 10)))
        cmd = mission.update(VehicleState(GroundPoint(0, 0), 0, 0, 0), None, 1)
        self.assertEqual(cmd.state, 'STALE_TELEMETRY')
        self.assertEqual((cmd.north_m_s, cmd.east_m_s), (0, 0))
