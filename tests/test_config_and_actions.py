import pytest
import yaml

from gesture_control.actions import ActionDispatcher, RecordingBackend
from gesture_control.config import (
    ConfigError,
    default_config_text,
    load_config,
    parse_config,
)
from gesture_control.engine import Event


def raw_default():
    return yaml.safe_load(default_config_text())


def test_default_config_loads(config):
    assert config.gestures["open_palm"].action == "play_pause"
    assert config.mouse_mode.toggle_gesture == "three"
    for binding in config.gestures.values():
        assert binding.action in config.actions


def test_config_from_file(tmp_path):
    path = tmp_path / "g.yaml"
    raw = raw_default()
    raw["gestures"]["open_palm"]["action"] = "screenshot"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).gestures["open_palm"].action == "screenshot"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["gestures"].update(wave={"action": "play_pause"}), "unknown gesture"),
        (lambda r: r["gestures"].update(fist={"action": "nope"}), "not defined"),
        (lambda r: r["actions"].update(x={"type": "teleport"}), "unknown type"),
        (lambda r: r["actions"].update(x={"type": "hotkey"}), "needs keys"),
        (lambda r: r["actions"].update(x={"type": "launch"}), "needs targets"),
        (lambda r: r["gestures"].update(three={"action": "play_pause"}), "mouse mode"),
        (lambda r: r["swipe"].update(speed=3), "unknown keys"),
        (lambda r: r.update(extra=1), "top-level"),
        (lambda r: r["stabilizer"].update(min_agree=9), "min_agree"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message):
    raw = raw_default()
    mutate(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_dispatcher_runs_each_action_type(config):
    backend = RecordingBackend()
    d = ActionDispatcher(config, backend)
    assert d.dispatch(Event("open_palm", 0.0)) == "play_pause"
    assert d.dispatch(Event("swipe_right", 0.0)) == "next_window"
    assert d.dispatch(Event("scroll", 0.0, -3)) == "scroll"
    assert d.dispatch(Event("cursor", 0.0, (0.25, 0.75))) == "move_cursor"
    assert d.dispatch(Event("pinch_start", 0.0)) == "mouse_down"
    assert d.dispatch(Event("pinch_end", 0.0)) == "mouse_up"
    assert d.dispatch(Event("scribble", 0.0)) == "open_notes"
    assert d.dispatch(Event("mouse_mode", 0.0, True)) is None  # engine-internal, unbound
    assert backend.calls == [
        ("press", ("playpause",)),
        ("hotkey", ("alt", "tab")),
        ("scroll", -3),
        ("move_to", (0.25, 0.75)),
        ("mouse_down", None),
        ("mouse_up", None),
        ("launch", ("onenote:", "notepad.exe")),
    ]
    assert d.label("play_pause") == "Play/pause"
