"""Image-centering state machine only; no OpenCV, hardware or flight commands."""

import math

from .models import CenteringResult, NavigationDecision, NavigationState, Point


class Navigation:
    def __init__(self) -> None:
        self.state = NavigationState.IDLE

    def update(self, centering: CenteringResult | None) -> NavigationDecision:
        """Missing/invalid input safely resets to IDLE; unseen green means search."""
        x = y = 0.0
        if centering is None:
            self.state = NavigationState.IDLE
        elif not centering.detected:
            self.state = NavigationState.SEARCHING_REFERENCE
        else:
            errors = (centering.normalized_error_x, centering.normalized_error_y)
            if any(value is None or not math.isfinite(value) or abs(value) > 1 for value in errors):
                self.state = NavigationState.IDLE
            else:
                x, y = errors
                self.state = NavigationState.CENTERED if centering.centered else NavigationState.CENTERING
        return NavigationDecision(self.state, x, y)


def centering_from_dict(data: dict) -> CenteringResult:
    """Accept a standalone CenteringResult or the centering member of VisionResult."""
    data = data.get("centering", data)
    if not isinstance(data, dict):
        raise ValueError("Centering input must be an object")
    if type(data.get("detected")) is not bool or type(data.get("centered")) is not bool:
        raise ValueError("detected and centered must be booleans")
    center = data.get("center")
    return CenteringResult(
        detected=data["detected"],
        center=Point(**center) if center is not None else None,
        error_x=data.get("error_x"), error_y=data.get("error_y"),
        normalized_error_x=data.get("normalized_error_x"),
        normalized_error_y=data.get("normalized_error_y"), centered=data["centered"],
    )
