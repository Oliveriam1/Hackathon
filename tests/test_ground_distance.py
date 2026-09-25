"""Pure geometry plus compatibility with red_tracker's actual projection convention."""

import math
import pytest

from ground_distance import ViewingAngles, distance_from_angles, from_red_detection, pixel_to_viewing_angles
from src.models import CameraIntrinsics, Point

K = CameraIntrinsics(200, 200, 320, 240, 640, 480)


@pytest.mark.parametrize("pixel,right,forward", [(Point(320,240),0,0), (Point(420,240),5,0),
                                                (Point(220,240),-5,0), (Point(320,140),0,5),
                                                (Point(320,340),0,-5), (Point(420,140),5,5)])
def test_pixel_directions(pixel, right, forward):
    angle = pixel_to_viewing_angles(pixel, K, gimbal_right_deg=0, gimbal_forward_deg=0, roll_deg=0, pitch_deg=0)
    result = distance_from_angles(angle, 10)
    assert result.right_offset_m == pytest.approx(right, abs=1e-10)
    assert result.forward_offset_m == pytest.approx(forward, abs=1e-10)
    assert result.ground_distance_m == pytest.approx(math.hypot(right, forward))
    assert result.slant_distance_m**2 == pytest.approx(100 + right**2 + forward**2)


@pytest.mark.parametrize("ax,ay", [(30,0), (0,30), (25,-20), (-40,35)])
def test_matches_red_tracker_gimbal_geometry(ax, ay):
    import red_tracker
    angles = pixel_to_viewing_angles(Point(320,240), K, gimbal_right_deg=ax,
                                     gimbal_forward_deg=ay, roll_deg=0, pitch_deg=0)
    result = distance_from_angles(angles, 10)
    right, forward = red_tracker.aim_to_ground(ax, ay, 10)
    assert (result.right_offset_m, result.forward_offset_m) == pytest.approx((right, forward))


def test_off_axis_projection_roundtrip_with_red_tracker():
    import red_tracker as rt
    # Invoke only the projection mathematics: no camera, servo or simulation loop.
    camera = object.__new__(rt.SimCamera)
    state = dict(x=30, y=15, alt=20, yaw=55)
    target = (31, 17)
    pixel = camera._project([target], state, (12, -8))[0]
    intrinsics = CameraIntrinsics(rt.F_X, rt.F_Y, rt.CAPTURE_W/2, rt.CAPTURE_H/2, rt.CAPTURE_W, rt.CAPTURE_H)
    result = from_red_detection(dict(x=pixel[0], y=pixel[1], angle=80, tilt=70), intrinsics=intrinsics,
                                frame_size=(rt.CAPTURE_W, rt.CAPTURE_H), altitude_agl_m=20,
                                gimbal_right_deg=12, gimbal_forward_deg=-8, roll_deg=0, pitch_deg=0)
    assert rt.body_to_field(result.right_offset_m, result.forward_offset_m, state) == pytest.approx(target)


@pytest.mark.parametrize("roll,pitch,right,forward", [(30,0,-10/math.sqrt(3),0), (0,30,0,10/math.sqrt(3))])
def test_attitude(roll, pitch, right, forward):
    angles = pixel_to_viewing_angles(Point(320,240), K, gimbal_right_deg=0,
                                     gimbal_forward_deg=0, roll_deg=roll, pitch_deg=pitch)
    result = distance_from_angles(angles, 10)
    assert (result.right_offset_m,result.forward_offset_m) == pytest.approx((right,forward), abs=1e-10)


def test_angle_distance_and_sine_relationship():
    result = distance_from_angles(ViewingAngles(45, 90), 10)
    assert result.ground_distance_m == pytest.approx(10)
    assert result.slant_distance_m * math.sin(math.pi/4) == pytest.approx(10)
    assert distance_from_angles(ViewingAngles(0,None), 10).ground_distance_m == 0


@pytest.mark.parametrize("height", [0,-1,float('nan'),float('inf')])
def test_invalid_agl(height):
    with pytest.raises(ValueError):
        distance_from_angles(ViewingAngles(30,0), height)


@pytest.mark.parametrize("angle", [-1,90,100,float('nan'),float('inf'),89.999999999])
def test_invalid_angles(angle):
    with pytest.raises(ValueError):
        ViewingAngles(angle,0)


def test_missing_invalid_detection_and_horizon():
    kwargs = dict(intrinsics=K, frame_size=(640,480), altitude_agl_m=10,
                  gimbal_right_deg=0, gimbal_forward_deg=0, roll_deg=0, pitch_deg=0)
    assert from_red_detection(None, **kwargs) is None
    for detection in ({}, {'x':float('nan'),'y':240}, {'x':640,'y':240}):
        with pytest.raises(ValueError):
            from_red_detection(detection, **kwargs)
    with pytest.raises(ValueError, match='resolution'):
        from_red_detection(dict(x=320,y=240), **{**kwargs, 'frame_size':(320,240)})
    with pytest.raises(ValueError, match='horizon'):
        from_red_detection(dict(x=320,y=240), **{**kwargs, 'gimbal_right_deg':90})
