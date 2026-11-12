"""Temporal filters: pose stabilization (debounces flicker) and One Euro point smoothing."""

from __future__ import annotations

import math
from collections import Counter, deque

from gesture_control.classifier import Pose


class PoseStabilizer:
    """Majority vote over a sliding window with hysteresis.

    The stable pose only changes when a different pose wins at least `min_agree` of the last
    `window` frames, so one or two misclassified frames never change it.
    """

    def __init__(self, window: int = 5, min_agree: int = 3) -> None:
        if not 1 <= min_agree <= window:
            raise ValueError("need 1 <= min_agree <= window")
        self.window = window
        self.min_agree = min_agree
        self._history: deque[Pose] = deque(maxlen=window)
        self.stable: Pose = Pose.NONE

    def update(self, pose: Pose) -> Pose:
        self._history.append(pose)
        winner, count = Counter(self._history).most_common(1)[0]
        if winner != self.stable and count >= self.min_agree:
            self.stable = winner
        return self.stable

    def reset(self) -> None:
        self._history.clear()
        self.stable = Pose.NONE


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012): low lag when moving, low jitter when still.

    `min_cutoff` (Hz) sets smoothing at rest; `beta` raises the cutoff with speed.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = self._alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x

    def reset(self) -> None:
        self._x, self._t, self._dx = None, None, 0.0


class PointFilter:
    """One Euro filter applied independently to x and y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.02) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta)
        self._fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self._fx(x, t), self._fy(y, t)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
