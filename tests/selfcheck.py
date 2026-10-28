"""selfcheck.py - M0 headless smoke tests.

Usage: python3 tests/selfcheck.py   (from the repo root)

Fixtures over cameras: everything here runs without hardware.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capture import SyntheticSource, FpsCounter   # noqa: E402
from src.hud import HUDState, draw_hud               # noqa: E402
from src.main import main                            # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<38} {detail}")


def t_synthetic_shape():
    src = SyntheticSource(640, 360)
    ok, frame = src.read()
    good = ok and frame.shape == (360, 640, 3) and frame.dtype == np.uint8
    check("synthetic source frame shape/dtype", good, str(frame.shape))


def t_synthetic_deterministic():
    a = SyntheticSource(320, 240).read()[1]
    b = SyntheticSource(320, 240).read()[1]
    check("synthetic source deterministic", bool(np.array_equal(a, b)))


def t_hud_draws():
    src = SyntheticSource(640, 360)
    frame = src.read()[1]
    before = frame.copy()
    draw_hud(frame, HUDState(fps=29.7))
    changed = int(np.count_nonzero(frame != before))
    good = frame.shape == before.shape and changed > 1000
    check("HUD overlay draws onto frame", good, f"{changed} px changed")


def t_hud_with_step_state():
    frame = SyntheticSource(640, 360).read()[1]
    s = HUDState(step_index=2, step_count=6, instruction="Place rack",
                 gesture="OPEN_PALM", verification="4/4 tubes", locked=True)
    draw_hud(frame, s)
    check("HUD renders populated state", True, "no exceptions")


def t_headless_loop():
    rc = main(["--synthetic", "--headless", "--frames", "60",
               "--width", "640", "--height", "360"])
    check("headless capture loop runs 60 frames", rc == 0)


def t_fps_counter():
    src = SyntheticSource(640, 360, fps=30.0)
    fps = FpsCounter()
    for _ in range(60):
        src.read()
        measured = fps.tick()
    good = measured >= 15.0
    check("synthetic pacing near 30 fps", good, f"{measured:.1f} fps")


if __name__ == "__main__":
    t_synthetic_shape()
    t_synthetic_deterministic()
    t_hud_draws()
    t_hud_with_step_state()
    t_headless_loop()
    t_fps_counter()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks PASS")
    sys.exit(0 if all(RESULTS) else 1)
