"""Replay every landmark fixture through the engine and dispatcher and check what would fire."""

from collections import Counter

import pytest
from conftest import FIXTURE_PATHS

from gesture_control.actions import ActionDispatcher, RecordingBackend
from gesture_control.engine import GestureEngine
from gesture_control.fixtures import load_recording

CONTINUOUS = {"cursor", "scroll", "mouse_mode"}


def replay(path, config):
    rec = load_recording(path)
    engine = GestureEngine(config)
    backend = RecordingBackend()
    dispatcher = ActionDispatcher(config, backend)
    discrete, scroll, modes, cursor_moves = [], 0, [], 0
    for t, hand in rec.frames:
        for event in engine.process(hand, t).events:
            action = dispatcher.dispatch(event)
            if event.name == "scroll":
                scroll += event.value
            elif event.name == "cursor":
                cursor_moves += 1
            elif event.name == "mouse_mode":
                modes.append(event.value)
            elif action:
                discrete.append(action)
    return rec, discrete, scroll, modes, cursor_moves, backend


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=lambda p: p.stem)
def test_fixture_fires_expected_actions(path, config):
    rec, discrete, scroll, modes, cursor_moves, _ = replay(path, config)
    exp = rec.expected
    assert exp, f"{path.name} has no expectations"

    if "actions" in exp:
        assert discrete == exp["actions"]
    if "action_counts" in exp:
        counts = Counter(discrete)
        assert set(counts) == set(exp["action_counts"])
        for action, (lo, hi) in exp["action_counts"].items():
            assert lo <= counts[action] <= hi, (action, counts[action])
    if "scroll" in exp:
        assert discrete == []
        assert scroll < 0 if exp["scroll"] == "down" else scroll > 0
    else:
        assert scroll == 0
    assert modes == exp.get("mouse_mode", [])
    assert (cursor_moves > 0) == exp.get("cursor", False)


def test_mouse_session_backend_calls(config):
    path = next(p for p in FIXTURE_PATHS if p.stem == "mouse_session")
    *_, backend = replay(path, config)
    kinds = [kind for kind, _ in backend.calls]
    assert kinds.index("mouse_down") < kinds.index("mouse_up")
    moves = [v for kind, v in backend.calls if kind == "move_to"]
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in moves)
    # The hand moved right and up, so the cursor should too.
    assert moves[-1][0] > moves[0][0]
    assert moves[-1][1] < moves[0][1]


def test_fixtures_have_unique_names():
    names = [load_recording(p).name for p in FIXTURE_PATHS]
    assert len(names) == len(set(names))
