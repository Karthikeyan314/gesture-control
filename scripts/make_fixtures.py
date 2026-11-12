"""Generate the synthetic landmark fixtures in tests/fixtures.

A 2-D hand skeleton with MediaPipe's 21-landmark layout and realistic proportions is posed per
gesture, placed in the frame, rotated, moved along a path and given Gaussian jitter, then
sampled at 30 fps. Glitch frames (a wrong pose) and dropped frames (no hand) are mixed in where a
test needs them. Real recordings made with `gesture-control record` use the same JSON format and
can sit next to these.

    uv run python scripts/make_fixtures.py
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path

from gesture_control.fixtures import Recording
from gesture_control.landmarks import Hand

FPS = 30
OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

Vec = tuple[float, float]

# Hand-local units: wrist at the origin, palm length (wrist to middle knuckle) = 1, fingers
# pointing up (negative y, as in image coordinates), thumb on the left.
MCP: dict[str, Vec] = {
    "index": (-0.30, -0.95),
    "middle": (0.0, -1.0),
    "ring": (0.25, -0.93),
    "pinky": (0.47, -0.82),
}
SEGMENTS: dict[str, tuple[float, float, float]] = {
    "index": (0.45, 0.27, 0.22),
    "middle": (0.50, 0.30, 0.24),
    "ring": (0.46, 0.28, 0.22),
    "pinky": (0.36, 0.22, 0.20),
}
THUMB: dict[str, list[Vec]] = {  # CMC, MCP, IP, TIP
    "extended": [(-0.25, -0.25), (-0.50, -0.40), (-0.72, -0.52), (-0.92, -0.62)],
    "curled": [(-0.25, -0.25), (-0.40, -0.48), (-0.30, -0.68), (-0.10, -0.72)],
    "pinch": [(-0.25, -0.25), (-0.50, -0.55), (-0.60, -0.95), (-0.62, -1.33)],
}
INDEX_PINCH: list[Vec] = [(-0.40, -1.35), (-0.52, -1.50), (-0.60, -1.40)]  # PIP, DIP, TIP

# finger states per pose: thumb, index, middle, ring, pinky  (E extended, C curled, P pinch)
POSES: dict[str, str] = {
    "open_palm": "EEEEE",
    "fist": "CCCCC",
    "thumbs_up": "ECCCC",
    "thumbs_down": "ECCCC",
    "point": "CECCC",
    "peace": "CEECC",
    "three": "CEEEC",
    "call": "ECCCE",
    "pinch": "PPCCC",
}


def _unit(v: Vec) -> Vec:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def _add(a: Vec, b: Vec, k: float = 1.0) -> Vec:
    return (a[0] + k * b[0], a[1] + k * b[1])


def _finger(name: str, state: str) -> list[Vec]:
    mcp = MCP[name]
    if state == "P":
        return [mcp, *INDEX_PINCH]
    d = _unit(mcp)
    prox, mid, dist = SEGMENTS[name]
    if state == "E":
        pip = _add(mcp, d, prox)
        dip = _add(pip, d, mid)
        tip = _add(dip, d, dist)
    else:  # curled: proximal phalanx foreshortened, the rest folds back toward the palm
        pip = _add(mcp, d, prox * 0.5)
        dip = _add(pip, d, -mid * 0.6)
        tip = _add(dip, d, -dist * 0.5)
    return [mcp, pip, dip, tip]


def skeleton(pose: str) -> list[Vec]:
    states = POSES[pose]
    thumb_key = {"E": "extended", "C": "curled", "P": "pinch"}[states[0]]
    pts: list[Vec] = [(0.0, 0.0), *THUMB[thumb_key]]
    for name, s in zip(("index", "middle", "ring", "pinky"), states[1:], strict=True):
        pts.extend(_finger(name, s))
    return pts


def _rotation_for(pose: str) -> float:
    """Extra rotation so thumbs-up/down poses point the thumb straight up or down."""
    if pose not in ("thumbs_up", "thumbs_down"):
        return 0.0
    tmcp, tip = THUMB["extended"][1], THUMB["extended"][3]
    current = math.atan2(tip[1] - tmcp[1], tip[0] - tmcp[0])
    target = -math.pi / 2 if pose == "thumbs_up" else math.pi / 2
    return target - current


def place(
    pose: str, x: float, y: float, rng: random.Random, scale=0.16, tilt=0.0, noise=0.003
) -> Hand:
    angle = _rotation_for(pose) + tilt
    c, s = math.cos(angle), math.sin(angle)
    points = []
    for px, py in skeleton(pose):
        rx, ry = c * px - s * py, s * px + c * py
        points.append(
            (
                x + scale * rx + rng.gauss(0, noise),
                y + scale * ry + rng.gauss(0, noise),
                0.0,
            )
        )
    return Hand(points=tuple(points), handedness="Right", score=0.97)


Path2D = Callable[[float], Vec]  # u in [0, 1] -> wrist position


def still(x: float, y: float) -> Path2D:
    return lambda u: (x, y)


def line(a: Vec, b: Vec) -> Path2D:
    return lambda u: (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


class Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.frames: list[tuple[float, Hand | None]] = []
        self.t = 0.0

    def _tick(self) -> None:
        self.t = round(self.t + 1 / FPS, 4)

    def none(self, seconds: float) -> Builder:
        for _ in range(round(seconds * FPS)):
            self.frames.append((self.t, None))
            self._tick()
        return self

    def pose(
        self, pose: str, seconds: float, path: Path2D, tilt: float = 0.0, noise: float = 0.003
    ) -> Builder:
        n = round(seconds * FPS)
        for i in range(n):
            x, y = path(i / max(n - 1, 1))
            self.frames.append((self.t, place(pose, x, y, self.rng, tilt=tilt, noise=noise)))
            self._tick()
        return self

    def scribble(self, seconds: float, x: float, y: float, amp: float, hz: float) -> Builder:
        n = round(seconds * FPS)
        for i in range(n):
            t = i / FPS
            hand = place("point", x + amp * math.sin(2 * math.pi * hz * t), y, self.rng)
            self.frames.append((self.t, hand))
            self._tick()
        return self

    def glitch(self, every: int, pose: str) -> Builder:
        """Replace every n-th frame with a single wrong pose at the same place."""
        for i in range(every, len(self.frames), every):
            t, hand = self.frames[i]
            if hand is not None:
                w = hand.points[0]
                self.frames[i] = (t, place(pose, w[0], w[1], self.rng))
        return self

    def drop(self, *indices: int) -> Builder:
        for i in indices:
            self.frames[i] = (self.frames[i][0], None)
        return self


def fixtures() -> list[Recording]:
    rec: list[Recording] = []

    def add(name: str, description: str, builder: Builder, **expected) -> None:
        rec.append(Recording(name, builder.frames, description, "synthetic", expected))

    centre = still(0.5, 0.75)
    add(
        "open_palm_hold",
        "Open palm held still for 1.2 s: one play/pause.",
        Builder(1).none(0.3).pose("open_palm", 1.2, centre).none(0.3),
        actions=["play_pause"],
    )
    add(
        "open_palm_long_hold",
        "Open palm held for 3 s: still exactly one play/pause (no repeat).",
        Builder(2).pose("open_palm", 3.0, centre, tilt=0.15),
        actions=["play_pause"],
    )
    add(
        "open_palm_glitchy",
        "Open palm with a fist frame every 8 frames, two dropped frames and heavier jitter.",
        Builder(3).pose("open_palm", 1.5, centre, noise=0.006).glitch(8, "fist").drop(15, 16),
        actions=["play_pause"],
    )
    add(
        "thumbs_up_hold",
        "Thumbs up held 1.5 s: volume up, repeating.",
        Builder(4).pose("thumbs_up", 1.5, still(0.5, 0.6)),
        action_counts={"volume_up": [4, 6]},
    )
    add(
        "thumbs_down_hold",
        "Thumbs down held 1.5 s: volume down, repeating.",
        Builder(5).pose("thumbs_down", 1.5, still(0.5, 0.5)),
        action_counts={"volume_down": [4, 6]},
    )
    add(
        "swipe_right",
        "Palm rests, sweeps right across the frame, then rests: one next-window, no play/pause.",
        Builder(6)
        .pose("open_palm", 0.3, still(0.3, 0.75))
        .pose("open_palm", 0.25, line((0.3, 0.75), (0.72, 0.73)))
        .pose("open_palm", 0.8, still(0.72, 0.73)),
        actions=["next_window"],
    )
    add(
        "swipe_left",
        "Palm sweeps left: one previous-window.",
        Builder(7)
        .pose("open_palm", 0.3, still(0.7, 0.75))
        .pose("open_palm", 0.25, line((0.7, 0.75), (0.28, 0.76)))
        .pose("open_palm", 0.3, still(0.28, 0.76)),
        actions=["previous_window"],
    )
    add(
        "scroll_down",
        "Peace sign rests, moves down and stays: scrolls down.",
        Builder(8)
        .pose("peace", 0.3, still(0.5, 0.7))
        .pose("peace", 0.3, line((0.5, 0.7), (0.5, 0.82)))
        .pose("peace", 0.8, still(0.5, 0.82)),
        scroll="down",
    )
    add(
        "scroll_up",
        "Peace sign rests, moves up and stays: scrolls up.",
        Builder(9)
        .pose("peace", 0.3, still(0.5, 0.8))
        .pose("peace", 0.3, line((0.5, 0.8), (0.5, 0.68)))
        .pose("peace", 0.8, still(0.5, 0.68)),
        scroll="up",
    )
    add(
        "mic_grip_hold",
        "Closed fist (mic grip) held 1.6 s: opens the recorder once.",
        Builder(10).pose("fist", 1.6, still(0.5, 0.7)),
        actions=["start_recording"],
    )
    add(
        "scribble",
        "Pointing finger wiggled left-right at 3 Hz for 1.2 s: opens notes once.",
        Builder(11).pose("point", 0.3, still(0.5, 0.75)).scribble(1.2, 0.5, 0.75, 0.06, 3.0),
        actions=["open_notes"],
    )
    add(
        "point_still",
        "Pointing finger held still: nothing happens.",
        Builder(12).pose("point", 1.5, still(0.5, 0.75)),
        actions=[],
    )
    add(
        "call_hold",
        "Call-me sign held 1 s: one screenshot.",
        Builder(13).pose("call", 1.0, still(0.5, 0.7)),
        actions=["screenshot"],
    )
    add(
        "pose_transitions",
        "Quick changes between poses, none held long enough: nothing happens.",
        Builder(14)
        .pose("open_palm", 0.3, centre)
        .pose("fist", 0.3, centre)
        .pose("peace", 0.1, centre)
        .pose("call", 0.3, centre)
        .pose("open_palm", 0.3, centre),
        actions=[],
    )
    add(
        "no_hand",
        "Empty frames only.",
        Builder(15).none(1.0),
        actions=[],
    )
    add(
        "mouse_session",
        "Three fingers toggle mouse mode on, point moves the cursor, pinch presses and releases, "
        "three fingers toggle it off.",
        Builder(16)
        .pose("three", 1.0, still(0.45, 0.7))
        .pose("point", 0.6, line((0.45, 0.7), (0.6, 0.6)))
        .pose("pinch", 0.4, still(0.6, 0.6))
        .pose("point", 0.3, still(0.6, 0.6))
        .pose("three", 1.0, still(0.6, 0.6))
        .pose("open_palm", 0.3, still(0.6, 0.6)),
        actions=["mouse_down", "mouse_up"],
        mouse_mode=[True, False],
        cursor=True,
    )
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        if '"source":"synthetic"' in old.read_text("utf-8"):
            old.unlink()
    for r in fixtures():
        r.save(OUT / f"{r.name}.json")
        print(f"wrote {r.name}.json ({len(r.frames)} frames)")


if __name__ == "__main__":
    main()
