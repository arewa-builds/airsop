"""gestures.py - rule-based gesture classifier + hold-to-trigger debouncer (M2).

Six-gesture vocabulary shared with SterileView:
    open_palm, fist, pinch, point_left, point_right, peace

Why geometric rules instead of a learned model: every decision is
explainable in one line ("index tip is 1.3x farther from the wrist than its
PIP joint, so the index is extended"), there is no training data to
collect, and the thresholds can be audited. In a regulated lab that
matters more than a few points of accuracy.

All tests are ratios of 2D distances, so they are invariant to hand size,
distance from the camera, and in-plane rotation. z is ignored: MediaPipe
depth is the noisiest coordinate, and gloves make it worse.

Debouncing: a gesture fires only after it is held for `hold_frames`
consecutive frames (default 18, about 0.6 s at 30 fps), fires once, and
must be released before it can fire again. Short dropouts (the tracker
losing a gloved hand for a frame or two) do not reset the hold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hands import (WRIST, THUMB_TIP, INDEX_MCP, MIDDLE_MCP, INDEX_TIP,
                    HandObservation)

GESTURES = ("open_palm", "fist", "pinch", "point_left", "point_right",
            "peace")
NONE = "none"

# AirSOP meaning of each gesture. M3 (SOP engine) consumes these actions.
ACTIONS = {
    "point_right": "next",
    "point_left": "back",
    "pinch": "confirm",
    "fist": "lock",          # toggles lock; the only gesture honored while locked
    "open_palm": "pause",
    "peace": "help",
}

# (PIP, TIP) landmark pairs for index, middle, ring, pinky.
_FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))

EXTENDED_RATIO = 1.10     # |wrist->tip| / |wrist->pip| above this = extended
PINCH_RATIO = 0.25        # |thumb tip - index tip| / palm length below = pinch
POINT_AXIS = 1.4          # |dx| / |dy| of the index above this = horizontal


@dataclass
class Features:
    extended: tuple[bool, bool, bool, bool]
    ratios: tuple[float, float, float, float]
    pinch: float
    index_dir: tuple[float, float]


def features(lm: np.ndarray) -> Features:
    xy = np.asarray(lm, float)[:, :2]
    w = xy[WRIST]
    palm = np.linalg.norm(xy[MIDDLE_MCP] - w) + 1e-9
    ratios = tuple(float(np.linalg.norm(xy[t] - w) /
                         (np.linalg.norm(xy[p] - w) + 1e-9))
                   for p, t in _FINGERS)
    ext = tuple(r > EXTENDED_RATIO for r in ratios)
    pinch = float(np.linalg.norm(xy[THUMB_TIP] - xy[INDEX_TIP]) / palm)
    d = xy[INDEX_TIP] - xy[INDEX_MCP]
    d = d / (np.linalg.norm(d) + 1e-9)
    return Features(ext, ratios, pinch, (float(d[0]), float(d[1])))


def classify(obs: HandObservation | np.ndarray | None) -> str:
    """One frame of landmarks -> gesture label or 'none'."""
    if obs is None:
        return NONE
    lm = obs.landmarks if isinstance(obs, HandObservation) else obs
    f = features(lm)
    i, m, r, p = f.extended

    if f.pinch < PINCH_RATIO and m and r and p:
        return "pinch"
    if not (i or m or r or p):
        return "fist"
    if i and m and r and p:
        return "open_palm"
    if i and m and not r and not p:
        return "peace"
    if i and not m and not r and not p:
        dx, dy = f.index_dir
        if abs(dx) > POINT_AXIS * abs(dy):
            return "point_right" if dx > 0 else "point_left"
    return NONE


class HoldDebouncer:
    """Fire a gesture once after it is held; require release to re-arm.

    update() returns the gesture name on the frame it fires, else None.
    `progress` (0..1) drives the HUD hold ring.
    """

    def __init__(self, hold_frames: int = 18, grace_frames: int = 3):
        self.hold_frames = hold_frames
        self.grace_frames = grace_frames
        self.reset()

    def reset(self) -> None:
        self.candidate = NONE
        self.count = 0
        self.missed = 0
        self.fired = False

    @property
    def progress(self) -> float:
        if self.candidate == NONE or self.fired:
            return 0.0
        return min(1.0, self.count / self.hold_frames)

    def update(self, label: str) -> str | None:
        if label == self.candidate and label != NONE:
            self.count += 1
            self.missed = 0
        elif label == NONE and self.candidate != NONE \
                and self.missed < self.grace_frames:
            self.missed += 1          # brief dropout: keep the hold alive
            return None
        else:
            self.candidate = label
            self.count = 1 if label != NONE else 0
            self.missed = 0
            self.fired = False
        if not self.fired and self.candidate != NONE \
                and self.count >= self.hold_frames:
            self.fired = True
            return self.candidate
        return None
