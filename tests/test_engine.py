"""Engine behaviour that the fixtures don't pin down on their own."""

import dataclasses

from conftest import first_hand, recording

from gesture_control.classifier import Pose, classify
from gesture_control.config import Binding
from gesture_control.engine import GestureEngine

DT = 1 / 30


def run(engine, hand, seconds, t0=0.0):
    events, t = [], t0
    for _ in range(round(seconds / DT)):
        events += engine.process(hand, t).events
        t += DT
    return events, t


def names(events):
    return [e.name for e in events]


def test_hold_fires_once_while_held(config):
    events, _ = run(GestureEngine(config), first_hand("open_palm_hold"), 4.0)
    assert names(events) == ["open_palm"]


def test_cooldown_blocks_quick_refire(config):
    engine = GestureEngine(config)
    palm = first_hand("open_palm_hold")
    events, t = run(engine, palm, 1.0)  # fires at about 0.57 s
    assert names(events) == ["open_palm"]
    _, t = run(engine, None, 0.2, t)  # hand away briefly
    events, _ = run(engine, palm, 0.7, t)  # held again: would fire at about 1.77 s
    assert names(events) == []  # inside the 1.5 s cooldown


def test_cooldown_expires(config):
    engine = GestureEngine(config)
    palm = first_hand("open_palm_hold")
    _, t = run(engine, palm, 1.0)
    _, t = run(engine, None, 1.0, t)
    events, _ = run(engine, palm, 1.0, t)
    assert names(events) == ["open_palm"]


def test_repeat_binding(config):
    cfg = dataclasses.replace(
        config, gestures={**config.gestures, "fist": Binding("volume_up", hold=0.2, repeat=0.1)}
    )
    events, _ = run(GestureEngine(cfg), first_hand("mic_grip_hold"), 1.0)
    assert 6 <= names(events).count("fist") <= 9


def test_stable_pose_lags_by_min_agree_frames(config):
    engine = GestureEngine(config)
    palm = first_hand("open_palm_hold")
    poses = [engine.process(palm, i * DT).pose for i in range(5)]
    agree = config.stabilizer.min_agree
    assert poses[: agree - 1] == [Pose.NONE] * (agree - 1)
    assert poses[agree - 1 :] == [Pose.OPEN_PALM] * (5 - agree + 1)


def test_hand_lost_releases_pinch(config):
    engine = GestureEngine(config)
    _, t = run(engine, first_hand("mouse_session"), 1.0)  # three fingers: mouse mode on
    assert engine.mouse_mode
    pinch = next(h for _, h in recording("mouse_session").frames if h and classify(h) is Pose.PINCH)
    events, t = run(engine, pinch, 0.3, t)
    assert "pinch_start" in names(events)
    events, _ = run(engine, None, 0.3, t)
    assert names(events) == ["pinch_end"]


def test_progress_reported_while_holding(config):
    engine = GestureEngine(config)
    fist = first_hand("mic_grip_hold")
    progress = [engine.process(fist, i * DT).hold_progress for i in range(40)]
    assert progress[0] == 0.0
    assert 0.3 < progress[18] < 0.7
    assert progress[-1] == 1.0
