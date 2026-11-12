"""Load and validate the gesture-to-action YAML config."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from gesture_control.classifier import ClassifierConfig, Pose
from gesture_control.motion import ScribbleConfig, ScrollConfig, SwipeConfig

# Events the engine can emit. Static poses fire after being held; the rest are dynamic.
POSE_EVENTS = frozenset(p.value for p in Pose if p is not Pose.NONE)
DYNAMIC_EVENTS = frozenset(
    {"swipe_left", "swipe_right", "scroll", "scribble", "cursor", "pinch_start", "pinch_end"}
)
KNOWN_EVENTS = POSE_EVENTS | DYNAMIC_EVENTS

ACTION_TYPES = frozenset(
    {"key", "hotkey", "launch", "scroll", "move_cursor", "mouse_down", "mouse_up", "click"}
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Binding:
    action: str
    hold: float = 0.0  # seconds a static pose must be held still before firing
    repeat: float | None = None  # seconds between repeats while still held (None = once)
    cooldown: float = 0.0  # minimum seconds between two firings


@dataclass(frozen=True)
class ActionSpec:
    type: str
    keys: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()  # for launch: tried in order until one opens
    label: str = ""


@dataclass(frozen=True)
class StabilizerConfig:
    window: int = 5
    min_agree: int = 3


@dataclass(frozen=True)
class MouseModeConfig:
    toggle_gesture: str = "three"
    hold: float = 0.8
    cooldown: float = 1.5
    margin: float = 0.15  # camera border ignored so the screen edges are reachable
    min_cutoff: float = 1.5
    beta: float = 8.0


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True


@dataclass(frozen=True)
class ModelConfig:
    path: str = "models/hand_landmarker.task"
    url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5


@dataclass(frozen=True)
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    swipe: SwipeConfig = field(default_factory=SwipeConfig)
    scribble: ScribbleConfig = field(default_factory=ScribbleConfig)
    scroll: ScrollConfig = field(default_factory=ScrollConfig)
    mouse_mode: MouseModeConfig = field(default_factory=MouseModeConfig)
    gestures: dict[str, Binding] = field(default_factory=dict)
    actions: dict[str, ActionSpec] = field(default_factory=dict)


def _build(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{section}: unknown keys {sorted(unknown)}")
    try:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    except TypeError as exc:
        raise ConfigError(f"{section}: {exc}") from exc


def parse_config(raw: dict[str, Any]) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown top-level keys {sorted(unknown)}")

    actions = {
        name: _build(ActionSpec, spec, f"actions.{name}")
        for name, spec in (raw.get("actions") or {}).items()
    }
    gestures = {
        name: _build(Binding, spec, f"gestures.{name}")
        for name, spec in (raw.get("gestures") or {}).items()
    }
    cfg = Config(
        camera=_build(CameraConfig, raw.get("camera"), "camera"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        classifier=_build(ClassifierConfig, raw.get("classifier"), "classifier"),
        stabilizer=_build(StabilizerConfig, raw.get("stabilizer"), "stabilizer"),
        swipe=_build(SwipeConfig, raw.get("swipe"), "swipe"),
        scribble=_build(ScribbleConfig, raw.get("scribble"), "scribble"),
        scroll=_build(ScrollConfig, raw.get("scroll"), "scroll"),
        mouse_mode=_build(MouseModeConfig, raw.get("mouse_mode"), "mouse_mode"),
        gestures=gestures,
        actions=actions,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    for name, spec in cfg.actions.items():
        if spec.type not in ACTION_TYPES:
            raise ConfigError(f"actions.{name}: unknown type {spec.type!r}")
        if spec.type in {"key", "hotkey"} and not spec.keys:
            raise ConfigError(f"actions.{name}: {spec.type} needs keys")
        if spec.type == "launch" and not spec.targets:
            raise ConfigError(f"actions.{name}: launch needs targets")
    for name, binding in cfg.gestures.items():
        if name not in KNOWN_EVENTS:
            raise ConfigError(f"gestures.{name}: unknown gesture (known: {sorted(KNOWN_EVENTS)})")
        if binding.action not in cfg.actions:
            raise ConfigError(f"gestures.{name}: action {binding.action!r} is not defined")
        if name == cfg.mouse_mode.toggle_gesture:
            raise ConfigError(f"gestures.{name}: already used to toggle mouse mode")
    if cfg.mouse_mode.toggle_gesture not in POSE_EVENTS:
        raise ConfigError(f"mouse_mode.toggle_gesture: {cfg.mouse_mode.toggle_gesture!r}")
    s = cfg.stabilizer
    if not 1 <= s.min_agree <= s.window:
        raise ConfigError("stabilizer: need 1 <= min_agree <= window")


def default_config_text() -> str:
    return resources.files("gesture_control").joinpath("default_config.yaml").read_text("utf-8")


def load_config(path: str | Path | None = None) -> Config:
    text = Path(path).read_text("utf-8") if path else default_config_text()
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    return parse_config(raw)
