import math

import pytest
from conftest import first_hand, recording

from gesture_control import landmarks as lm
from gesture_control.classifier import (
    ClassifierConfig,
    Pose,
    classify,
    finger_states,
    pinch_distance,
)
from gesture_control.landmarks import Hand

POSE_FIXTURES = {
    "open_palm_hold": Pose.OPEN_PALM,
    "mic_grip_hold": Pose.FIST,
    "thumbs_up_hold": Pose.THUMBS_UP,
    "thumbs_down_hold": Pose.THUMBS_DOWN,
    "point_still": Pose.POINT,
    "scroll_down": Pose.PEACE,
    "call_hold": Pose.CALL,
}


@pytest.mark.parametrize(("fixture", "pose"), POSE_FIXTURES.items())
def test_every_frame_of_a_held_pose_classifies(fixture, pose):
    frames = [h for _, h in recording(fixture).frames if h is not None]
    assert frames
    assert {classify(h) for h in frames} == {pose}


def test_no_hand_is_none():
    assert classify(None) is Pose.NONE


def test_finger_states_open_palm_and_fist():
    cfg = ClassifierConfig()
    assert finger_states(first_hand("open_palm_hold"), cfg).as_tuple() == (True,) * 5
    assert finger_states(first_hand("mic_grip_hold"), cfg).as_tuple() == (False,) * 5


def test_pinch_is_detected_in_mouse_session():
    frames = [h for _, h in recording("mouse_session").frames if h is not None]
    assert Pose.PINCH in {classify(h) for h in frames}


def test_fist_is_not_a_pinch_even_with_close_fingertips():
    # In a fist the thumb tip rests near the index tip, but the index is curled into the palm.
    fist = first_hand("mic_grip_hold")
    assert pinch_distance(fist) < 0.5
    assert classify(fist) is Pose.FIST


def _transform(hand: Hand, scale: float, angle: float, dx: float, dy: float) -> Hand:
    wx, wy, _ = hand[lm.WRIST]
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for x, y, z in hand.points:
        rx, ry = x - wx, y - wy
        pts.append((wx + dx + scale * (c * rx - s * ry), wy + dy + scale * (s * rx + c * ry), z))
    return Hand(points=tuple(pts))


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.6])
@pytest.mark.parametrize("angle", [-0.4, 0.0, 0.4])
def test_poses_survive_distance_and_tilt(scale, angle):
    # Thumbs up/down are excluded: their meaning depends on absolute orientation.
    for fixture, pose in POSE_FIXTURES.items():
        if pose in (Pose.THUMBS_UP, Pose.THUMBS_DOWN):
            continue
        hand = _transform(first_hand(fixture), scale, angle, 0.0, -0.1)
        assert classify(hand) is pose, (fixture, scale, angle)


def test_thumb_sideways_is_not_thumbs_up():
    thumbs_up = first_hand("thumbs_up_hold")
    sideways = _transform(thumbs_up, 1.0, math.pi / 2, 0.0, 0.0)
    assert classify(sideways) is Pose.NONE


def test_hand_requires_21_points():
    with pytest.raises(ValueError):
        Hand(points=((0.0, 0.0, 0.0),) * 20)
