"""Gesture engine: turns a stream of (hand, timestamp) into debounced gesture events.

The engine is pure and clock-free (timestamps are passed in), so the same code runs on the live
webcam feed and on recorded landmark fixtures in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gesture_control import landmarks as lm
from gesture_control.classifier import Pose, classify
from gesture_control.config import Config
from gesture_control.landmarks import Hand
from gesture_control.motion import (
    Cooldown,
    HoldTrigger,
    ScribbleDetector,
    ScrollJoystick,
    SwipeDetector,
)
from gesture_control.smoothing import PointFilter, PoseStabilizer

CURSOR_POSES = (Pose.POINT, Pose.PINCH)


@dataclass(frozen=True)
class Event:
    name: str
    t: float
    value: Any = None


@dataclass
class FrameResult:
    t: float
    hand: Hand | None
    raw_pose: Pose
    pose: Pose
    mouse_mode: bool
    events: list[Event] = field(default_factory=list)
    hold_progress: float = 0.0  # 0..1 toward the current pose's hold action
    cursor: tuple[float, float] | None = None


class GestureEngine:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.stabilizer = PoseStabilizer(config.stabilizer.window, config.stabilizer.min_agree)
        self.swipe = SwipeDetector(config.swipe)
        self.scribble = ScribbleDetector(config.scribble)
        self.scroll = ScrollJoystick(config.scroll)
        mm = config.mouse_mode
        self.cursor_filter = PointFilter(mm.min_cutoff, mm.beta)
        self.mode_hold = HoldTrigger(mm.hold)
        self.holds: dict[Pose, HoldTrigger] = {
            Pose(name): HoldTrigger(b.hold, b.repeat)
            for name, b in config.gestures.items()
            if name in Pose._value2member_map_
        }
        self.cooldown = Cooldown()
        self.mouse_mode = False
        self._pinch_down = False
        self._hold_blocked = False  # set after a swipe until the pose changes

    # -- public -----------------------------------------------------------------------------

    def process(self, hand: Hand | None, t: float) -> FrameResult:
        raw = classify(hand, self.cfg.classifier)
        previous = self.stabilizer.stable
        pose = self.stabilizer.update(raw)
        events: list[Event] = []

        if pose != previous:
            self._on_pose_change(previous, pose, t, events)

        result = FrameResult(t=t, hand=hand, raw_pose=raw, pose=pose, mouse_mode=self.mouse_mode)
        if hand is None:
            self.swipe.reset()
            result.events = self._debounce(events)
            return result

        wrist = hand[lm.WRIST]
        self._mode_toggle(pose, wrist, t, events)
        if self.mouse_mode:
            result.cursor = self._cursor(hand, pose, t, events)
        else:
            self._dynamic(hand, raw, pose, t, events)
            result.hold_progress = self._static_hold(pose, wrist, t, events)

        if pose.value == self.cfg.mouse_mode.toggle_gesture:
            result.hold_progress = self.mode_hold.progress(t)
        result.mouse_mode = self.mouse_mode
        result.events = self._debounce(events)
        return result

    def reset(self, t: float) -> list[Event]:
        """Forget all state (used when pausing). Returns a pinch release if a button is down."""
        events: list[Event] = []
        if self._pinch_down:
            self._release_pinch(t, events)
        self.stabilizer.reset()
        for trigger in (*self.holds.values(), self.mode_hold):
            trigger.reset()
        self.swipe.reset()
        self.scribble.reset()
        self.scroll.reset()
        self.cursor_filter.reset()
        self.mouse_mode = False
        self._hold_blocked = False
        return events

    # -- internals --------------------------------------------------------------------------

    def _on_pose_change(self, old: Pose, new: Pose, t: float, events: list[Event]) -> None:
        for trigger in self.holds.values():
            trigger.reset()
        self.mode_hold.reset()
        self._hold_blocked = False
        if old == Pose.POINT:
            self.scribble.reset()
        if old == Pose.PEACE:
            self.scroll.reset()
        if self._pinch_down and new != Pose.PINCH:
            self._release_pinch(t, events)
        if self.mouse_mode and new == Pose.PINCH:
            self._pinch_down = True
            events.append(Event("pinch_start", t))
        if new not in CURSOR_POSES:
            self.cursor_filter.reset()

    def _release_pinch(self, t: float, events: list[Event]) -> None:
        self._pinch_down = False
        events.append(Event("pinch_end", t))

    def _mode_toggle(self, pose: Pose, wrist: lm.Point, t: float, events: list[Event]) -> None:
        mm = self.cfg.mouse_mode
        if pose.value != mm.toggle_gesture:
            return
        if self.mode_hold.update(t, wrist[0], wrist[1]) and self.cooldown.ready(
            "mouse_mode", t, mm.cooldown
        ):
            self.mouse_mode = not self.mouse_mode
            if not self.mouse_mode and self._pinch_down:
                self._release_pinch(t, events)
            self.swipe.reset()
            self.scribble.reset()
            self.scroll.reset()
            events.append(Event("mouse_mode", t, self.mouse_mode))

    def _cursor(
        self, hand: Hand, pose: Pose, t: float, events: list[Event]
    ) -> tuple[float, float] | None:
        if pose not in CURSOR_POSES:
            return None
        # The knuckle barely moves when pinching, unlike the fingertip, so clicks land where aimed.
        x, y, _ = hand[lm.INDEX_MCP]
        m = self.cfg.mouse_mode.margin
        span = 1.0 - 2 * m
        sx = min(max((x - m) / span, 0.0), 1.0)
        sy = min(max((y - m) / span, 0.0), 1.0)
        cursor = self.cursor_filter(sx, sy, t)
        events.append(Event("cursor", t, cursor))
        return cursor

    def _dynamic(self, hand: Hand, raw: Pose, pose: Pose, t: float, events: list[Event]) -> None:
        wrist = hand[lm.WRIST]
        # Track swipes on the raw pose too: motion blur makes fast palms flicker.
        if Pose.OPEN_PALM in (raw, pose):
            direction = self.swipe.update(t, wrist[0], wrist[1])
            if direction:
                events.append(Event(f"swipe_{direction}", t))
                self._hold_blocked = True
                for trigger in self.holds.values():
                    trigger.reset()
        else:
            self.swipe.reset()

        if pose == Pose.POINT and self.scribble.update(t, hand[lm.INDEX_TIP][0]):
            events.append(Event("scribble", t))

        if pose == Pose.PEACE:
            y = (hand[lm.INDEX_TIP][1] + hand[lm.MIDDLE_TIP][1]) / 2
            clicks = self.scroll.update(t, y)
            if clicks:
                events.append(Event("scroll", t, clicks))

    def _static_hold(self, pose: Pose, wrist: lm.Point, t: float, events: list[Event]) -> float:
        trigger = self.holds.get(pose)
        if trigger is None or self._hold_blocked:
            return 0.0
        if trigger.update(t, wrist[0], wrist[1]):
            events.append(Event(pose.value, t))
        return trigger.progress(t)

    def _debounce(self, events: list[Event]) -> list[Event]:
        kept = []
        for e in events:
            binding = self.cfg.gestures.get(e.name)
            if (
                binding
                and binding.cooldown > 0
                and not self.cooldown.ready(e.name, e.t, binding.cooldown)
            ):
                continue
            kept.append(e)
        return kept
