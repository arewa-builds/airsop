"""hud.py - heads-up display overlay for AirSOP.

M0: static chrome (banner, step panel, status chips, footer).
M1: palm-anchored cursor ring and a hand-presence chip.
M2: hold-progress arc on the cursor and the last fired action.
Later milestones only mutate HUDState fields; the layout stays stable, the
same way a good instrument panel keeps its gauges in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (235, 235, 235)          # primary text, BGR
MUTED = (150, 150, 150)
PANEL = (24, 24, 24)
ACCENT = (200, 160, 40)        # teal-ish accent in BGR
WARN = (60, 60, 200)


@dataclass
class HUDState:
    """Everything the overlay is allowed to know. Defaults = M0 skeleton."""
    step_index: int | None = None        # None = no protocol loaded
    step_count: int | None = None
    instruction: str = "(no protocol loaded)"
    gesture: str = "--"
    verification: str = "--"
    fps: float = 0.0
    locked: bool = False
    cursor: tuple[int, int] | None = None   # px, EMA-smoothed palm center
    hand: bool = False                      # hand detected this frame
    skeleton: list[tuple[int, int]] | None = None   # 21 landmark px
    hold: float = 0.0                       # 0..1 hold-to-trigger progress
    last_action: str = ""                   # e.g. "pinch -> confirm"
    milestone: str = "M2 gestures"
    notes: list[str] = field(default_factory=list)


def _panel(frame, x0, y0, x1, y1, alpha=0.65):
    roi = frame[y0:y1, x0:x1]
    fill = np.full_like(roi, PANEL)
    frame[y0:y1, x0:x1] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)


def _text(frame, text, org, scale=0.6, color=INK, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


# MediaPipe hand topology: thumb, four fingers, palm.
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
         (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


def draw_hud(frame: np.ndarray, state: HUDState) -> np.ndarray:
    """Draw the overlay in place and return the frame."""
    h, w = frame.shape[:2]

    # top banner
    _panel(frame, 0, 0, w, 56)
    _text(frame, "AIRSOP", (16, 38), 0.9, INK, 2)
    _text(frame, "touchless bench workstation", (150, 38), 0.6, MUTED)
    if state.locked:
        _text(frame, "LOCKED", (w - 120, 38), 0.7, WARN, 2)

    # step panel (left)
    _panel(frame, 12, 72, 388, 168)
    if state.step_index is None:
        _text(frame, "Step -- / --", (28, 108), 0.7, MUTED)
    else:
        _text(frame, f"Step {state.step_index} / {state.step_count}",
              (28, 108), 0.7, INK)
    _text(frame, state.instruction[:44], (28, 140), 0.55, MUTED)

    # status chips (right)
    _panel(frame, w - 268, 72, w - 12, 168)
    _text(frame, f"gesture:  {state.gesture}", (w - 252, 108), 0.55, INK)
    _text(frame, f"verify:   {state.verification}", (w - 252, 140), 0.55, INK)
    _text(frame, "hand" if state.hand else "no hand", (w - 252, 160), 0.45,
          ACCENT if state.hand else MUTED)

    # hand skeleton (thin, so it never competes with the cursor)
    if state.skeleton is not None:
        for a, b in BONES:
            cv2.line(frame, state.skeleton[a], state.skeleton[b], MUTED, 1,
                     cv2.LINE_AA)
        for pt in state.skeleton:
            cv2.circle(frame, pt, 2, INK, -1, cv2.LINE_AA)

    # cursor: ring + center dot at the smoothed palm center
    if state.cursor is not None:
        cx, cy = state.cursor
        color = ACCENT if state.hand else MUTED
        cv2.circle(frame, (cx, cy), 18, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, color, -1, cv2.LINE_AA)
        if state.hold > 0.0:
            cv2.ellipse(frame, (cx, cy), (26, 26), -90, 0,
                        int(360 * state.hold), INK, 3, cv2.LINE_AA)

    if state.last_action:
        _panel(frame, 12, 176, 388, 210)
        _text(frame, state.last_action, (28, 200), 0.55, ACCENT)

    # footer
    _panel(frame, 0, h - 44, w, h)
    _text(frame, f"{state.fps:4.1f} fps", (16, h - 15), 0.55, MUTED)
    _text(frame, f"{w}x{h}", (130, h - 15), 0.55, MUTED)
    _text(frame, state.milestone, (w - 200, h - 15), 0.55, ACCENT)
    return frame
