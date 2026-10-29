"""make_fixtures.py - build the M2 landmark replay fixtures.

Usage: python tools/make_fixtures.py          (from the repo root)

Each fixture is JSONL (JSON Lines): one meta line, then one line per frame
with 21 normalized landmarks (or null when no hand is tracked) and the
ground-truth label. The selfcheck replays every tests/fixtures/*.jsonl
through classify() + HoldDebouncer and compares the fired events with
meta.expected_events.

These fixtures are synthetic (procedural poses plus jitter, drift,
rotation and dropouts), and deterministic, so the selfcheck can also prove
the committed files match this generator. Real gloved recordings captured
with `python -m src.main --record-landmarks ...` use the same format and
are replayed the same way.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.poses import hand_pose  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _segments_to_frames(segments, seed, jitter=0.003, rotation=0.0,
                        scale=0.18, left=False, drift=True):
    """segments: list of (label, n_frames, opts). label None = no hand."""
    rng = np.random.default_rng(seed)
    frames, n = [], 0
    for label, count, opts in segments:
        for _ in range(count):
            t = n / 30.0
            c = (0.5 + (0.12 * np.sin(0.6 * t) if drift else 0.0),
                 0.55 + (0.06 * np.cos(0.4 * t) if drift else 0.0))
            if label is None or opts.get("dropout"):
                frames.append({"label": label or "none", "landmarks": None})
            else:
                pose = "point_left" if label == "point_up" else label
                rot = rotation + opts.get("rotation", 0.0)
                if label == "point_up":      # index straight up: not a command
                    rot += 90.0
                lm = hand_pose(pose, rotation=rot, center=c,
                               scale=opts.get("scale", scale),
                               left=opts.get("left", left))
                lm[:, :2] += rng.normal(0.0, jitter, (21, 2))
                truth = "none" if label == "point_up" else label
                frames.append({"label": truth,
                               "landmarks": np.round(lm, 4).tolist()})
            n += 1
    return frames


def scenarios():
    H = 30                                  # comfortable hold, 1 s
    gap = (None, 10, {})
    out = {}

    out["each_gesture_once"] = dict(
        description="Each of the six gestures held 1 s, separated by no-hand gaps.",
        expected=["open_palm", "fist", "pinch", "point_left", "point_right",
                  "peace"],
        frames=_segments_to_frames(
            [gap] + [s for g in ["open_palm", "fist", "pinch", "point_left",
                                 "point_right", "peace"]
                     for s in ((g, H, {}), gap)], seed=1))

    out["flicker_rejected"] = dict(
        description="Pinch and point shown for 10 frames each (under the 18-frame hold): nothing fires.",
        expected=[],
        frames=_segments_to_frames(
            [gap, ("pinch", 10, {}), gap, ("point_right", 10, {}), gap],
            seed=2))

    out["dropout_tolerated"] = dict(
        description="point_right held with two 2-frame tracking dropouts: fires once.",
        expected=["point_right"],
        frames=_segments_to_frames(
            [gap, ("point_right", 8, {}), ("point_right", 2, {"dropout": 1}),
             ("point_right", 6, {}), ("point_right", 2, {"dropout": 1}),
             ("point_right", 12, {}), gap], seed=3))

    out["long_hold_fires_once"] = dict(
        description="Fist held for 4 s: fires exactly once, no auto-repeat.",
        expected=["fist"],
        frames=_segments_to_frames([gap, ("fist", 120, {}), gap], seed=4))

    out["release_rearms"] = dict(
        description="Pinch, release for 6 frames, pinch again: two confirms.",
        expected=["pinch", "pinch"],
        frames=_segments_to_frames(
            [gap, ("pinch", 25, {}), (None, 6, {}), ("pinch", 25, {}), gap],
            seed=5))

    out["non_command_pose_ignored"] = dict(
        description="Index pointing straight up (not in the vocabulary) for 2 s: nothing fires.",
        expected=[],
        frames=_segments_to_frames([gap, ("point_up", 60, {}), gap], seed=6))

    out["protocol_walk_left_hand_tilted"] = dict(
        description=("A bench run: next, confirm, next, confirm, back, help, "
                     "pause. Left hand, 15 degree tilt, smaller in frame, "
                     "heavier jitter."),
        expected=["point_right", "pinch", "point_right", "pinch",
                  "point_left", "peace", "open_palm"],
        frames=_segments_to_frames(
            [gap] + [s for g in ["point_right", "pinch", "point_right",
                                 "pinch", "point_left", "peace", "open_palm"]
                     for s in ((g, 24, {}), (None, 8, {}))],
            seed=7, jitter=0.004, rotation=15.0, scale=0.12, left=True))
    return out


def render(name, sc) -> str:
    meta = {"type": "meta", "name": name, "source": "synthetic", "fps": 30,
            "description": sc["description"],
            "expected_events": sc["expected"], "frames": len(sc["frames"])}
    lines = [json.dumps(meta)]
    for i, f in enumerate(sc["frames"]):
        lines.append(json.dumps({"type": "frame", "i": i, **f},
                                separators=(",", ":")))
    return "\n".join(lines) + "\n"


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, sc in scenarios().items():
        path = FIXTURES / f"{name}.jsonl"
        path.write_text(render(name, sc))
        print(f"wrote {path.relative_to(FIXTURES.parent.parent)} "
              f"({len(sc['frames'])} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
