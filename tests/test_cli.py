import pytest
from conftest import FIXTURES, first_hand

from gesture_control.app import main
from gesture_control.config import load_config
from gesture_control.engine import GestureEngine


def test_replay_prints_events(capsys):
    main(["replay", str(FIXTURES / "swipe_right.json")])
    out = capsys.readouterr().out
    assert "pose  open_palm" in out
    assert "event swipe_right -> next_window" in out


def test_init_config_round_trips(tmp_path):
    path = tmp_path / "gestures.yaml"
    main(["init-config", str(path)])
    assert load_config(path).gestures["scribble"].action == "open_notes"
    with pytest.raises(SystemExit):
        main(["init-config", str(path)])  # refuses to overwrite without --force
    main(["init-config", str(path), "--force"])


def test_engine_reset_releases_held_pinch(config):
    from conftest import recording

    from gesture_control.classifier import Pose, classify

    engine = GestureEngine(config)
    three = first_hand("mouse_session")
    pinch = next(h for _, h in recording("mouse_session").frames if h and classify(h) is Pose.PINCH)
    t = 0.0
    for _ in range(40):
        engine.process(three, t)
        t += 1 / 30
    for _ in range(10):
        engine.process(pinch, t)
        t += 1 / 30
    assert [e.name for e in engine.reset(t)] == ["pinch_end"]
    assert not engine.mouse_mode
    assert engine.reset(t) == []
