"""Draw the live overlay: skeleton, detected pose, hold progress, mode and last action."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from gesture_control.classifier import Pose
from gesture_control.engine import FrameResult
from gesture_control.landmarks import HAND_CONNECTIONS

GREEN = (90, 200, 90)
AMBER = (0, 180, 255)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
BLUE = (255, 160, 60)
PANEL = (30, 30, 30)

POSE_LABELS = {
    Pose.NONE: "no gesture",
    Pose.OPEN_PALM: "open palm",
    Pose.FIST: "fist (mic)",
    Pose.THUMBS_UP: "thumbs up",
    Pose.THUMBS_DOWN: "thumbs down",
    Pose.POINT: "point",
    Pose.PEACE: "peace (scroll)",
    Pose.THREE: "three fingers",
    Pose.PINCH: "pinch",
    Pose.CALL: "call me",
}


@dataclass
class OverlayState:
    last_action: str = ""
    last_action_t: float = -10.0
    fps: float = 0.0
    paused: bool = False


def _text(img, text, org, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw(frame, result: FrameResult, state: OverlayState) -> None:
    """Draw in place on a BGR frame."""
    h, w = frame.shape[:2]
    if result.hand is not None:
        pts = [(int(x * w), int(y * h)) for x, y, _ in result.hand.points]
        colour = BLUE if result.mouse_mode else GREEN
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], colour, 2)
        for p in pts:
            cv2.circle(frame, p, 3, WHITE, -1)

    # top panel
    cv2.rectangle(frame, (0, 0), (w, 64), PANEL, -1)
    if state.paused:
        _text(frame, "PAUSED  (p to resume)", (10, 26), 0.7, AMBER, 2)
    else:
        _text(frame, POSE_LABELS.get(result.pose, result.pose.value), (10, 26), 0.75, WHITE, 2)
    mode = "mouse mode" if result.mouse_mode else "gesture mode"
    _text(frame, mode, (w - 150, 26), 0.55, BLUE if result.mouse_mode else GREY)
    _text(frame, f"{state.fps:4.1f} fps", (w - 150, 52), 0.5, GREY)

    # hold progress bar
    if 0.0 < result.hold_progress < 1.0 and not state.paused:
        cv2.rectangle(frame, (10, 40), (210, 52), GREY, 1)
        cv2.rectangle(frame, (10, 40), (10 + int(200 * result.hold_progress), 52), AMBER, -1)

    # last action, fading out after two seconds
    age = result.t - state.last_action_t
    if state.last_action and age < 2.0:
        _text(frame, state.last_action, (10, h - 16), 0.9, GREEN, 2)

    if result.cursor is not None:
        cx, cy = result.cursor
        _text(frame, f"cursor {cx:.2f}, {cy:.2f}", (w - 210, h - 16), 0.5, BLUE)

    _text(frame, "p pause  h hide  q quit", (10, h - 44), 0.45, GREY)
