"""MediaPipe Tasks hand landmarker wrapper (imported only by the live app)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from gesture_control.config import ModelConfig
from gesture_control.landmarks import Hand

log = logging.getLogger(__name__)


def ensure_model(cfg: ModelConfig) -> Path:
    path = Path(cfg.path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hand landmarker model to %s", path)
    try:
        urllib.request.urlretrieve(cfg.url, path)  # noqa: S310 - fixed https URL from config
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise SystemExit(
            f"Could not download the model ({exc}).\n"
            f"Download it manually:\n  curl.exe -L -o {path} {cfg.url}"
        ) from exc
    return path


class HandDetector:
    """Runs the landmarker in VIDEO mode (uses tracking between frames, so it's cheaper)."""

    def __init__(self, cfg: ModelConfig) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(cfg))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_tracking_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, rgb_frame, t: float) -> Hand | None:
        """`rgb_frame` is an HxWx3 uint8 RGB array; `t` is seconds (monotonic)."""
        ms = max(int(t * 1000), self._last_ms + 1)  # timestamps must strictly increase
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, ms)
        if not result.hand_landmarks:
            return None
        points = tuple((p.x, p.y, p.z) for p in result.hand_landmarks[0])
        handed = result.handedness[0][0] if result.handedness else None
        return Hand(
            points=points,
            handedness=handed.category_name if handed else "Right",
            score=handed.score if handed else 1.0,
        )

    def close(self) -> None:
        self._landmarker.close()
