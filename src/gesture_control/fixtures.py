"""Landmark recordings: JSON files of timestamped hand landmarks, replayable without a webcam.

Format::

    {
      "name": "open_palm_hold",
      "description": "...",
      "source": "synthetic" | "recorded",
      "expected": {"actions": ["play_pause"], ...},   # optional, used by tests
      "frames": [{"t": 0.0, "hand": {"handedness": "Right", "score": 1.0,
                                     "landmarks": [[x, y, z], ...21]}}, {"t": 0.033, "hand": null}]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gesture_control.landmarks import Hand


@dataclass
class Recording:
    name: str
    frames: list[tuple[float, Hand | None]]
    description: str = ""
    source: str = "recorded"
    expected: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "expected": self.expected,
            "frames": [
                {"t": round(t, 4), "hand": hand.to_dict() if hand else None}
                for t, hand in self.frames
            ],
        }

    def save(self, path: str | Path) -> None:
        text = json.dumps(self.to_dict(), separators=(",", ":")) + "\n"
        Path(path).write_text(text, "utf-8", newline="\n")  # same bytes on every OS


def load_recording(path: str | Path) -> Recording:
    data = json.loads(Path(path).read_text("utf-8"))
    frames = [
        (float(f["t"]), Hand.from_dict(f["hand"]) if f.get("hand") else None)
        for f in data["frames"]
    ]
    return Recording(
        name=data.get("name", Path(path).stem),
        frames=frames,
        description=data.get("description", ""),
        source=data.get("source", "recorded"),
        expected=data.get("expected", {}),
    )
