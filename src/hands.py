"""hands.py - hand landmark tracking for AirSOP (M1).

HandTracker wraps the MediaPipe Hand Landmarker (Tasks API). The import is
lazy, so headless environments without mediapipe still run everything
else. SyntheticHandTracker implements the same interface with
deterministic procedural landmarks: a palm orbiting the synthetic reticle
with jitter, so the full loop (frame -> hands -> cursor -> HUD) runs with
no camera and no GPU.

Cursor anchor: the palm center (mean of wrist + four MCP joints) is far
more stable than any fingertip, which is why the cursor anchors there
instead of on the index tip. A gloved fingertip is also the landmark most
likely to wobble.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# MediaPipe landmark indices used by AirSOP-family gesture code.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 5, 9, 13, 17
INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 8, 12, 16, 20
PALM_JOINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / \
    "hand_landmarker.task"


@dataclass
class HandObservation:
    """One frame of one hand. landmarks: (21, 3) normalized x, y, z."""
    landmarks: np.ndarray
    handedness: str = "Right"

    @property
    def palm(self) -> np.ndarray:
        """Normalized (x, y) palm center: the cursor anchor."""
        pts = self.landmarks[list(PALM_JOINTS), :2]
        return pts.mean(axis=0)

    def palm_px(self, width: int, height: int) -> tuple[int, int]:
        p = self.palm
        return int(p[0] * width), int(p[1] * height)


def resolve_model(path: str | os.PathLike | None = None,
                  download: bool = True) -> Path:
    """Find the Hand Landmarker model, downloading it once if missing.

    Lookup order: explicit path, AIRSOP_HAND_MODEL env var, models/.
    The Docker image bakes the model in at build time, so containers run
    offline.
    """
    p = Path(path or os.environ.get("AIRSOP_HAND_MODEL") or DEFAULT_MODEL)
    if p.exists():
        return p
    if not download:
        raise FileNotFoundError(f"hand model not found at {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.replace(p)
    return p


class HandTracker:
    """Real hand tracking via the MediaPipe Hand Landmarker (VIDEO mode).

    Import is lazy on purpose: the selfcheck must pass without mediapipe.
    """

    def __init__(self, max_num_hands: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5,
                 model_path: str | os.PathLike | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise RuntimeError(
                "mediapipe is required for live hand tracking; "
                "pip install -r requirements-camera.txt, "
                "or run with --synthetic") from e
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(resolve_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray) -> HandObservation | None:
        import cv2
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=rgb)
        # VIDEO mode needs strictly increasing timestamps.
        ms = max(int((time.monotonic() - self._t0) * 1000), self._last_ms + 1)
        self._last_ms = ms
        res = self._lm.detect_for_video(image, ms)
        if not res.hand_landmarks:
            return None
        pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]],
                       dtype=np.float64)
        handed = "Right"
        if res.handedness and res.handedness[0]:
            handed = res.handedness[0][0].category_name
        return HandObservation(landmarks=pts, handedness=handed)

    def close(self) -> None:
        self._lm.close()


class SyntheticHandTracker:
    """Deterministic fake hand: palm orbits the reticle, plus jitter.

    Same interface as HandTracker. Jitter is realistic sensor noise the
    EMA cursor must absorb; the underlying orbit is ground truth.
    """

    # Procedural offsets from palm center for all 21 landmarks, normalized
    # units, roughly an open right palm pointing up. Only coherence matters.
    _POSE = np.array([
        [0.00, 0.06], [-0.03, 0.03], [-0.05, 0.00], [-0.06, -0.02],
        [-0.07, -0.04], [-0.02, -0.03], [-0.02, -0.06], [-0.02, -0.08],
        [-0.02, -0.10], [0.00, -0.03], [0.00, -0.07], [0.00, -0.09],
        [0.00, -0.11], [0.02, -0.03], [0.03, -0.06], [0.03, -0.08],
        [0.03, -0.10], [0.04, -0.02], [0.05, -0.04], [0.06, -0.06],
        [0.06, -0.08]], dtype=np.float64)

    def __init__(self, seed: int = 7, jitter: float = 0.004):
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter
        self._n = 0

    def truth(self, n: int) -> np.ndarray:
        """Noise-free orbit center at frame n (ground truth for tests)."""
        t = n / 30.0
        return np.array([0.5 + (1.0 / 3.0) * np.sin(t),
                         0.5 + 0.25 * np.cos(0.7 * t)])

    def process(self, frame_bgr: np.ndarray) -> HandObservation:
        c = self.truth(self._n) + self.rng.normal(0.0, self.jitter, 2)
        xy = self._POSE + c
        z = np.zeros((21, 1))
        self._n += 1
        return HandObservation(landmarks=np.hstack([xy, z]))

    def close(self) -> None:
        pass


class CursorEMA:
    """Exponential moving average cursor smoother.

    alpha trades responsiveness against jitter rejection: 0.4 settles in
    roughly five frames while absorbing landmark noise. First observation
    initializes the cursor directly, no ramp-in.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._p: np.ndarray | None = None

    def update(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        if self._p is None:
            self._p = point
        else:
            self._p = self.alpha * point + (1.0 - self.alpha) * self._p
        return self._p

    def reset(self) -> None:
        self._p = None

    @property
    def value(self) -> np.ndarray | None:
        return self._p
