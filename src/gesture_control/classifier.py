"""Rule-based static pose classification from landmark geometry.

Every rule works on distance ratios normalized by palm size, so it does not depend on how far the
hand is from the camera or how it is rotated in the image plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from gesture_control import landmarks as lm
from gesture_control.landmarks import Hand


class Pose(StrEnum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINT = "point"
    PEACE = "peace"
    THREE = "three"
    PINCH = "pinch"
    CALL = "call"


@dataclass(frozen=True)
class ClassifierConfig:
    # A long finger is extended when wrist→tip is this much longer than wrist→PIP.
    finger_extension_ratio: float = 1.15
    # The thumb is extended when its tip is this much farther from the pinky knuckle than its IP
    # joint is, and at least `thumb_min_spread` palm-lengths from the index knuckle.
    thumb_extension_ratio: float = 1.05
    thumb_min_spread: float = 0.5
    # Thumb tip must sit this many palm-lengths above/below the thumb knuckle for up/down.
    thumb_vertical_margin: float = 0.3
    # Pinch: thumb tip and index tip closer than this (palm-lengths) ...
    pinch_threshold: float = 0.25
    # ... with the index tip held out from the palm, which separates a pinch from a fist.
    pinch_min_index_reach: float = 1.1


@dataclass(frozen=True)
class FingerStates:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


def finger_extended(hand: Hand, finger: str, ratio: float) -> bool:
    _, pip, _, tip = lm.FINGERS[finger]
    return hand.distance(lm.WRIST, tip) > ratio * hand.distance(lm.WRIST, pip)


def thumb_extended(hand: Hand, cfg: ClassifierConfig) -> bool:
    away_from_palm = hand.distance(lm.THUMB_TIP, lm.PINKY_MCP) > cfg.thumb_extension_ratio * (
        hand.distance(lm.THUMB_IP, lm.PINKY_MCP)
    )
    spread = hand.distance(lm.THUMB_TIP, lm.INDEX_MCP) / hand.scale > cfg.thumb_min_spread
    return away_from_palm and spread


def finger_states(hand: Hand, cfg: ClassifierConfig) -> FingerStates:
    r = cfg.finger_extension_ratio
    return FingerStates(
        thumb=thumb_extended(hand, cfg),
        index=finger_extended(hand, "index", r),
        middle=finger_extended(hand, "middle", r),
        ring=finger_extended(hand, "ring", r),
        pinky=finger_extended(hand, "pinky", r),
    )


def pinch_distance(hand: Hand) -> float:
    return hand.distance(lm.THUMB_TIP, lm.INDEX_TIP) / hand.scale


def is_pinch(hand: Hand, cfg: ClassifierConfig) -> bool:
    index_reach = hand.distance(lm.WRIST, lm.INDEX_TIP) / hand.scale
    return pinch_distance(hand) < cfg.pinch_threshold and index_reach > cfg.pinch_min_index_reach


def thumb_vertical(hand: Hand, cfg: ClassifierConfig) -> int:
    """-1 when the thumb points down, +1 when it points up, 0 otherwise (image y grows down)."""
    dy = (hand[lm.THUMB_TIP][1] - hand[lm.THUMB_MCP][1]) / hand.scale
    if dy < -cfg.thumb_vertical_margin:
        return 1
    if dy > cfg.thumb_vertical_margin:
        return -1
    return 0


def classify(hand: Hand | None, cfg: ClassifierConfig | None = None) -> Pose:
    if hand is None:
        return Pose.NONE
    cfg = cfg or ClassifierConfig()
    if is_pinch(hand, cfg):
        return Pose.PINCH

    s = finger_states(hand, cfg)
    long_fingers = (s.index, s.middle, s.ring, s.pinky)

    # The thumb is the least reliable finger, so poses that do not need it ignore it.
    if all(long_fingers):
        return Pose.OPEN_PALM
    if not any(long_fingers):
        if not s.thumb:
            return Pose.FIST
        direction = thumb_vertical(hand, cfg)
        if direction > 0:
            return Pose.THUMBS_UP
        if direction < 0:
            return Pose.THUMBS_DOWN
        return Pose.NONE
    if long_fingers == (True, False, False, False):
        return Pose.POINT
    if long_fingers == (True, True, False, False):
        return Pose.PEACE
    if long_fingers == (True, True, True, False):
        return Pose.THREE
    if long_fingers == (False, False, False, True) and s.thumb:
        return Pose.CALL
    return Pose.NONE
