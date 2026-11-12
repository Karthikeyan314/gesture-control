"""Dynamic gesture detectors and trigger helpers. Positions are normalized image coordinates."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SwipeConfig:
    min_distance: float = 0.22  # fraction of frame width covered by the wrist
    max_duration: float = 0.6  # seconds
    max_vertical_ratio: float = 0.6  # |dy| / |dx| allowed


class SwipeDetector:
    """Fires 'left' or 'right' when the hand covers enough horizontal ground quickly."""

    def __init__(self, cfg: SwipeConfig | None = None) -> None:
        self.cfg = cfg or SwipeConfig()
        self._track: deque[tuple[float, float, float]] = deque()

    def update(self, t: float, x: float, y: float) -> str | None:
        self._track.append((t, x, y))
        while self._track and t - self._track[0][0] > self.cfg.max_duration:
            self._track.popleft()
        t0, x0, y0 = self._track[0]
        dx, dy = x - x0, y - y0
        if abs(dx) >= self.cfg.min_distance and abs(dy) <= self.cfg.max_vertical_ratio * abs(dx):
            self.reset()
            return "right" if dx > 0 else "left"
        return None

    def reset(self) -> None:
        self._track.clear()


@dataclass(frozen=True)
class ScribbleConfig:
    window: float = 1.5  # seconds of fingertip history examined
    min_reversals: int = 4  # direction changes needed
    min_amplitude: float = 0.04  # each stroke must travel this far (fraction of frame width)


class ScribbleDetector:
    """Detects a back-and-forth scribble of the fingertip (zigzag in x)."""

    def __init__(self, cfg: ScribbleConfig | None = None) -> None:
        self.cfg = cfg or ScribbleConfig()
        self._track: deque[tuple[float, float]] = deque()

    def update(self, t: float, x: float) -> bool:
        self._track.append((t, x))
        while self._track and t - self._track[0][0] > self.cfg.window:
            self._track.popleft()
        if count_reversals([p[1] for p in self._track], self.cfg.min_amplitude) >= (
            self.cfg.min_reversals
        ):
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._track.clear()


def count_reversals(values: list[float], min_amplitude: float) -> int:
    """Count direction changes in a 1-D signal, ignoring wiggles smaller than `min_amplitude`."""
    if not values:
        return 0
    reversals = 0
    direction = 0  # +1 rising, -1 falling, 0 not yet known
    lo = hi = extreme = values[0]
    for v in values[1:]:
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if v - lo >= min_amplitude:
                direction, extreme = 1, v
            elif hi - v >= min_amplitude:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= min_amplitude:
                reversals += 1
                direction, extreme = -1, v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= min_amplitude:
                reversals += 1
                direction, extreme = 1, v
    return reversals


@dataclass(frozen=True)
class ScrollConfig:
    deadzone: float = 0.04  # offset from the anchor (fraction of frame height) that does nothing
    gain: float = 120.0  # scroll clicks per second per unit offset beyond the deadzone
    max_rate: float = 30.0  # clicks per second cap


class ScrollJoystick:
    """Scroll like a joystick: the anchor is where the pose started; distance sets the speed.

    Moving the hand up from the anchor scrolls up (positive clicks), down scrolls down.
    """

    def __init__(self, cfg: ScrollConfig | None = None) -> None:
        self.cfg = cfg or ScrollConfig()
        self._anchor: float | None = None
        self._t: float | None = None
        self._carry = 0.0

    def update(self, t: float, y: float) -> int:
        if self._anchor is None or self._t is None:
            self._anchor, self._t = y, t
            return 0
        dt, self._t = t - self._t, t
        offset = self._anchor - y  # up is positive
        beyond = abs(offset) - self.cfg.deadzone
        if beyond <= 0:
            self._carry = 0.0
            return 0
        rate = min(self.cfg.gain * beyond, self.cfg.max_rate)
        self._carry += math.copysign(rate * dt, offset)
        clicks = int(self._carry)
        self._carry -= clicks
        return clicks

    def reset(self) -> None:
        self._anchor, self._t, self._carry = None, None, 0.0


class HoldTrigger:
    """Fires after a pose is held still for `hold` seconds, then every `repeat` seconds if set.

    Moving more than `tolerance` (fraction of frame) restarts the hold, so a swipe made with an
    open palm never also counts as holding the palm.
    """

    def __init__(self, hold: float, repeat: float | None = None, tolerance: float = 0.06) -> None:
        self.hold = hold
        self.repeat = repeat
        self.tolerance = tolerance
        self._start: float | None = None
        self._anchor: tuple[float, float] | None = None
        self._last_fire: float | None = None

    def update(self, t: float, x: float, y: float) -> bool:
        if self._start is None or self._anchor is None:
            self._start, self._anchor = t, (x, y)
        elif math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.tolerance:
            self._start, self._anchor, self._last_fire = t, (x, y), None
            return False
        if self._last_fire is None:
            if t - self._start >= self.hold:
                self._last_fire = t
                return True
            return False
        if self.repeat is not None and t - self._last_fire >= self.repeat:
            self._last_fire = t
            return True
        return False

    def progress(self, t: float) -> float:
        """0..1 while waiting for the first fire, 1 after it."""
        if self._start is None:
            return 0.0
        if self._last_fire is not None or self.hold <= 0:
            return 1.0
        return min((t - self._start) / self.hold, 1.0)

    def reset(self) -> None:
        self._start, self._anchor, self._last_fire = None, None, None


class Cooldown:
    """Per-event refractory period so an action can't fire again too soon."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def ready(self, name: str, t: float, cooldown: float) -> bool:
        last = self._last.get(name)
        if last is not None and t - last < cooldown:
            return False
        self._last[name] = t
        return True
