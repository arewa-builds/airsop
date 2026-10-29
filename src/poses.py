"""poses.py - procedural hand poses in MediaPipe's 21-landmark layout (M2).

A tiny 2D forward-kinematics hand: wrist at the origin, four MCP joints
fanned across the palm, three segments per finger. Each finger is either
extended (segments collinear) or curled (segments fold back toward the
palm). The whole hand can then be rotated, scaled, mirrored (left hand)
and translated.

Used by tools/make_fixtures.py to build replay fixtures and by
SyntheticHandTracker to script gestures in the synthetic demo. Image
coordinates: x right, y down, so "up" is negative y.
"""
from __future__ import annotations

import numpy as np

GESTURES = ("open_palm", "fist", "pinch", "point_left", "point_right",
            "peace")

# MCP offsets from the wrist (normalized hand units; palm length ~ 1.0).
_MCP = {"index": (-0.30, -0.95), "middle": (-0.05, -1.00),
        "ring": (0.18, -0.93), "pinky": (0.38, -0.82)}
_SEG = {"index": (0.40, 0.24, 0.20), "middle": (0.45, 0.28, 0.22),
        "ring": (0.42, 0.26, 0.20), "pinky": (0.32, 0.20, 0.18)}
_BASE = {"index": 0, "middle": 1, "ring": 2, "pinky": 3}
_FINGER_IDX = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

# Which fingers are extended for each gesture (index, middle, ring, pinky).
_EXTENDED = {
    "open_palm": (1, 1, 1, 1),
    "fist": (0, 0, 0, 0),
    "pinch": (0, 1, 1, 1),        # OK sign: thumb meets index tip
    "point": (1, 0, 0, 0),
    "peace": (1, 1, 0, 0),
}


def _rot(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def _finger(mcp, segs, up_dir, mode):
    """Return PIP, DIP, TIP for one finger. mode: ext, curl, or pinch."""
    bends = {"ext": (0, 0, 0), "curl": (0, 115, 75),
             "pinch": (25, 55, 45)}[mode]
    pts, total = [], 0.0
    p, d = np.array(mcp, float), np.array(up_dir, float)
    for seg, b in zip(segs, bends):
        total += b
        # cumulative bend toward the palm; the pinching index bends toward
        # the thumb side so its tip can meet the thumb tip
        dd = _rot(-total if mode == "pinch" else total) @ d
        p = p + seg * dd
        pts.append(p.copy())
    return pts


def hand_pose(gesture: str, *, rotation: float = 0.0, scale: float = 0.18,
              center=(0.5, 0.5), left: bool = False) -> np.ndarray:
    """21x3 landmarks for `gesture`. rotation in degrees, clockwise on screen.

    point_left / point_right rotate the hand so the index finger points
    along the screen axis; `rotation` adds on top of that.
    """
    if gesture not in GESTURES:
        raise ValueError(gesture)
    key = "point" if gesture.startswith("point") else gesture
    ext = _EXTENDED[key]
    lm = np.zeros((21, 2))
    up = np.array([0.0, -1.0])

    for name, (mx, my) in _MCP.items():
        i = _BASE[name]
        mcp = np.array([mx, my])
        if gesture == "pinch" and name == "index":
            mode = "pinch"
        else:
            mode = "ext" if ext[i] else "curl"
        # fan the fingers slightly outward from the middle finger
        fan = _rot(mx * 12.0) @ up
        pip, dip, tip = _finger(mcp, _SEG[name], fan, mode)
        a, b, c, d = _FINGER_IDX[name]
        lm[a], lm[b], lm[c], lm[d] = mcp, pip, dip, tip

    # thumb: CMC, MCP, IP, TIP (1..4)
    cmc = np.array([-0.30, -0.25])
    if gesture == "pinch":
        tip = lm[8] + np.array([0.0, 0.02])
        lm[1] = cmc
        lm[2] = cmc + (tip - cmc) * 0.40 + np.array([-0.08, 0.0])
        lm[3] = cmc + (tip - cmc) * 0.72 + np.array([-0.05, 0.0])
        lm[4] = tip
    elif gesture == "open_palm":
        th = np.array([-0.85, -0.50])
        th = th / np.linalg.norm(th)
        lm[1] = cmc
        lm[2] = cmc + 0.30 * th
        lm[3] = cmc + 0.55 * th
        lm[4] = cmc + 0.78 * th
    else:                                    # tucked across the palm
        lm[1] = cmc
        lm[2] = cmc + np.array([0.05, -0.22])
        lm[3] = cmc + np.array([0.20, -0.35])
        lm[4] = cmc + np.array([0.33, -0.42])

    base_rot = {"point_right": 90.0, "point_left": -90.0}.get(gesture, 0.0)
    if left:                                  # mirror image of a right hand
        lm[:, 0] *= -1.0
    lm = lm @ _rot(base_rot + rotation).T
    lm = lm * scale + np.asarray(center, float)
    return np.hstack([lm, np.zeros((21, 1))])
