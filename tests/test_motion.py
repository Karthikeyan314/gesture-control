import pytest

from gesture_control.classifier import Pose
from gesture_control.motion import (
    Cooldown,
    HoldTrigger,
    ScribbleDetector,
    ScrollJoystick,
    SwipeConfig,
    SwipeDetector,
    count_reversals,
)
from gesture_control.smoothing import OneEuroFilter, PoseStabilizer

DT = 1 / 30


def test_swipe_right_and_left():
    d = SwipeDetector(SwipeConfig(min_distance=0.2, max_duration=0.5))
    fired = [d.update(i * DT, 0.3 + 0.04 * i, 0.5) for i in range(8)]
    assert fired.count("right") == 1
    d.reset()
    fired = [d.update(i * DT, 0.7 - 0.04 * i, 0.5) for i in range(8)]
    assert fired.count("left") == 1


def test_slow_drift_is_not_a_swipe():
    d = SwipeDetector(SwipeConfig(min_distance=0.2, max_duration=0.5))
    assert not any(d.update(i * DT, 0.3 + 0.004 * i, 0.5) for i in range(90))


def test_diagonal_motion_is_not_a_swipe():
    d = SwipeDetector(SwipeConfig(min_distance=0.2, max_duration=0.5))
    assert not any(d.update(i * DT, 0.3 + 0.04 * i, 0.3 + 0.04 * i) for i in range(8))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0, 1, 0, 1, 0], 3),
        ([0, 0.01, 0, 0.01, 0], 0),  # wiggle below amplitude
        ([0, 0.5, 1.0], 0),  # monotonic
        ([], 0),
    ],
)
def test_count_reversals(values, expected):
    assert count_reversals(values, min_amplitude=0.1) == expected


def test_scribble_detector_needs_enough_reversals():
    d = ScribbleDetector()
    zig = [0.5 + (0.05 if (i // 4) % 2 else -0.05) for i in range(45)]
    assert any(d.update(i * DT, x) for i, x in enumerate(zig))
    d.reset()
    one_stroke = [0.4 + 0.01 * i for i in range(20)]
    assert not any(d.update(i * DT, x) for i, x in enumerate(one_stroke))


def test_scroll_joystick_direction_and_deadzone():
    j = ScrollJoystick()
    assert j.update(0.0, 0.5) == 0
    assert sum(j.update(i * DT, 0.52) for i in range(1, 30)) == 0  # inside the deadzone
    j.reset()
    j.update(0.0, 0.5)
    assert sum(j.update(i * DT, 0.6) for i in range(1, 30)) < 0  # hand lower: scroll down
    j.reset()
    j.update(0.0, 0.5)
    assert sum(j.update(i * DT, 0.4) for i in range(1, 30)) > 0


def test_hold_trigger_resets_on_movement():
    h = HoldTrigger(hold=0.5, tolerance=0.05)
    assert not any(h.update(i * DT, 0.5 + 0.01 * i, 0.5) for i in range(30))
    h.reset()
    assert any(h.update(i * DT, 0.5, 0.5) for i in range(30))


def test_cooldown():
    c = Cooldown()
    assert c.ready("a", 0.0, 1.0)
    assert not c.ready("a", 0.5, 1.0)
    assert c.ready("b", 0.5, 1.0)
    assert c.ready("a", 1.1, 1.0)


def test_stabilizer_ignores_single_glitches():
    s = PoseStabilizer(window=5, min_agree=3)
    seq = [Pose.OPEN_PALM] * 4 + [Pose.FIST] + [Pose.OPEN_PALM] * 3 + [Pose.NONE] + [Pose.OPEN_PALM]
    out = [s.update(p) for p in seq]
    assert out[2:] == [Pose.OPEN_PALM] * (len(seq) - 2)


def test_stabilizer_rejects_bad_args():
    with pytest.raises(ValueError):
        PoseStabilizer(window=3, min_agree=4)


def test_one_euro_reduces_jitter_but_follows_motion():
    f = OneEuroFilter(min_cutoff=1.0, beta=5.0)
    noisy = [0.5 + (0.01 if i % 2 else -0.01) for i in range(60)]
    out = [f(x, i * DT) for i, x in enumerate(noisy)]
    assert max(out[30:]) - min(out[30:]) < 0.01  # jitter amplitude was 0.02
    f.reset()
    ramp = [f(0.1 + 0.02 * i, i * DT) for i in range(40)]
    assert abs(ramp[-1] - (0.1 + 0.02 * 39)) < 0.05  # small lag on a fast move
