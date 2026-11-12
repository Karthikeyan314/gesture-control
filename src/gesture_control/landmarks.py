"""Hand landmark container, independent of MediaPipe so the gesture logic is testable anywhere.

Coordinates follow MediaPipe's normalized image convention: x and y are in [0, 1] relative to
the frame (y grows downwards), z is depth relative to the wrist (smaller is closer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float, float]

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# (mcp, pip, dip, tip) for the four long fingers.
FINGERS: dict[str, tuple[int, int, int, int]] = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Skeleton edges used by the overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip


@dataclass(frozen=True)
class Hand:
    """One detected hand: 21 landmarks plus the detector's handedness label and score."""

    points: tuple[Point, ...]
    handedness: str = "Right"
    score: float = 1.0

    def __post_init__(self) -> None:
        if len(self.points) != NUM_LANDMARKS:
            raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(self.points)}")

    def __getitem__(self, index: int) -> Point:
        return self.points[index]

    def distance(self, a: int, b: int) -> float:
        """Euclidean distance between two landmarks in the image plane (z is too noisy)."""
        pa, pb = self.points[a], self.points[b]
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    @property
    def scale(self) -> float:
        """Palm size (wrist to middle-finger knuckle); used to make thresholds distance-free."""
        return max(self.distance(WRIST, MIDDLE_MCP), 1e-6)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hand:
        points = tuple(
            (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
            for p in data["landmarks"]
        )
        return cls(
            points=points,
            handedness=data.get("handedness", "Right"),
            score=float(data.get("score", 1.0)),
        )

    def to_dict(self, precision: int = 4) -> dict[str, Any]:
        return {
            "handedness": self.handedness,
            "score": round(self.score, 3),
            "landmarks": [[round(c, precision) for c in p] for p in self.points],
        }
