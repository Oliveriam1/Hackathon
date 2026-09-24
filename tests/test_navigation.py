"""Navigation state transitions consume only shared centering data."""

import unittest

from src.models import CenteringResult, NavigationState, Point
from src.navigation import Navigation, centering_from_dict


class NavigationTests(unittest.TestCase):
    def test_missing_input_idle(self) -> None:
        self.assertEqual(Navigation().update(None).state, NavigationState.IDLE)

    def test_missing_green_searches(self) -> None:
        result = CenteringResult(False, None, None, None, None, None, False)
        decision = Navigation().update(result)
        self.assertEqual(decision.state, NavigationState.SEARCHING_REFERENCE)
        self.assertEqual(decision.horizontal_error, 0)

    def test_nonzero_errors_centering(self) -> None:
        result = CenteringResult(True, Point(10, 20), -13, 5, -0.13, 0.05, False)
        decision = Navigation().update(result)
        self.assertEqual(decision.state, NavigationState.CENTERING)
        self.assertEqual((decision.horizontal_error, decision.vertical_error), (-0.13, 0.05))

    def test_centered_then_lost_then_no_input(self) -> None:
        navigation = Navigation()
        centered = CenteringResult(True, Point(50, 50), 0, 0, 0, 0, True)
        self.assertEqual(navigation.update(centered).state, NavigationState.CENTERED)
        missing = CenteringResult(False, None, None, None, None, None, False)
        self.assertEqual(navigation.update(missing).state, NavigationState.SEARCHING_REFERENCE)
        self.assertEqual(navigation.update(None).state, NavigationState.IDLE)

    def test_invalid_error_is_idle(self) -> None:
        for error in (None, float("nan"), float("inf"), 2):
            result = CenteringResult(True, Point(1, 1), 1, 1, error, 0, False)
            self.assertEqual(Navigation().update(result).state, NavigationState.IDLE)

    def test_json_booleans_are_not_truthy_strings(self) -> None:
        with self.assertRaises(ValueError):
            centering_from_dict({"detected": "false", "centered": "true"})


if __name__ == "__main__":
    unittest.main()
