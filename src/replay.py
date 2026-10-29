"""replay.py - run landmark fixtures through the gesture pipeline (M2).

A fixture is JSONL: one meta line, then frames with 21 landmarks or null.
replay() feeds each frame through classify() and HoldDebouncer exactly as
the live loop does, and reports the fired events plus frame accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gestures import HoldDebouncer, classify


@dataclass
class ReplayResult:
    name: str
    expected: list[str]
    events: list[str]
    frame_acc: float           # labeled frames classified correctly
    frames: int

    @property
    def ok(self) -> bool:
        return self.events == self.expected


def load(path: str | Path) -> tuple[dict, list[dict]]:
    lines = Path(path).read_text().splitlines()
    meta = json.loads(lines[0])
    frames = [json.loads(x) for x in lines[1:] if x.strip()]
    return meta, frames


def replay(path: str | Path, hold_frames: int = 18) -> ReplayResult:
    meta, frames = load(path)
    deb = HoldDebouncer(hold_frames=hold_frames)
    events, hits, labeled = [], 0, 0
    for f in frames:
        lm = f.get("landmarks")
        label = classify(None if lm is None else np.asarray(lm))
        truth = f.get("label")
        if truth not in (None, "unlabeled") and lm is not None:
            labeled += 1
            hits += label == truth
        fired = deb.update(label)
        if fired:
            events.append(fired)
    acc = hits / labeled if labeled else float("nan")
    return ReplayResult(meta.get("name", Path(path).stem),
                        list(meta.get("expected_events", [])), events, acc,
                        len(frames))
