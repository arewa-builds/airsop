"""selfcheck.py - M0 + M1 headless checks.

Usage: python3 tests/selfcheck.py   (from the repo root)

Fixtures over cameras: everything here runs without hardware and without
mediapipe (the real tracker imports lazily; the synthetic one covers the
loop). If mediapipe is installed, one extra check loads the real Hand
Landmarker model; set AIRSOP_REQUIRE_MEDIAPIPE=1 (the Docker image does)
to make that check mandatory instead of SKIP.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.capture import SyntheticSource, FpsCounter   # noqa: E402
from src.hud import HUDState, draw_hud               # noqa: E402
from src.main import main                            # noqa: E402
from src.hands import (SyntheticHandTracker, HandObservation,  # noqa: E402
                       CursorEMA)

RESULTS = []
BLANK = np.zeros((4, 4, 3), np.uint8)


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name:<44} {detail}")


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
    check("headless capture+hand loop, 60 frames", rc == 0)


def t_fps_counter():
    src = SyntheticSource(640, 360, fps=30.0)
    fps = FpsCounter()
    for _ in range(60):
        src.read()
        measured = fps.tick()
    good = measured >= 15.0
    check("synthetic pacing near 30 fps", good, f"{measured:.1f} fps")


# ---- M1: hand tracking ------------------------------------------------

def t_synthetic_hand():
    h1, h2 = SyntheticHandTracker(), SyntheticHandTracker()
    s1 = [h1.process(BLANK).palm for _ in range(10)]
    s2 = [h2.process(BLANK).palm for _ in range(10)]
    check("synthetic hand deterministic",
          all(np.allclose(p, q) for p, q in zip(s1, s2)))
    obs = SyntheticHandTracker().process(BLANK)
    good = (obs.landmarks.shape == (21, 3)
            and np.all(obs.palm >= 0.0) and np.all(obs.palm <= 1.0))
    check("observation: 21 landmarks, palm normalized", good,
          f"palm={obs.palm.round(3)}")


def t_palm_anchor():
    lm = np.zeros((21, 3))
    lm[[0, 5, 9, 13, 17], :2] = 0.5      # palm joints at center
    lm[8, :2] = 0.9                       # index tip far away: must NOT move palm
    obs = HandObservation(landmarks=lm)
    px = obs.palm_px(640, 360)
    check("cursor anchors to palm center, not fingertip",
          np.allclose(obs.palm, 0.5) and px == (320, 180), str(px))


def t_ema():
    rng = np.random.default_rng(7)
    target = np.array([0.5, 0.5])
    raw = target + rng.normal(0.0, 0.01, (60, 2))
    cur = CursorEMA(alpha=0.4)
    out = np.array([cur.update(p) for p in raw])
    check("EMA reduces jitter", out[20:].std() < raw[20:].std(),
          f"raw std {raw[20:].std():.4f} -> smooth {out[20:].std():.4f}")
    check("EMA converges to target",
          np.all(np.abs(out[-1] - target) < 0.01), f"final={out[-1].round(4)}")


def t_cursor_tracks_truth():
    hand, cur = SyntheticHandTracker(), CursorEMA(alpha=0.4)
    errs = []
    for n in range(90):
        p = cur.update(hand.process(BLANK).palm)
        palm_offset = SyntheticHandTracker._POSE[[0, 5, 9, 13, 17]].mean(0)
        errs.append(np.linalg.norm(p - (hand.truth(n) + palm_offset)))
    mean_px = float(np.mean(errs[10:])) * 1280
    check("smoothed cursor tracks moving ground truth", mean_px < 40,
          f"mean err {mean_px:.1f} px @1280w")


def t_hud_cursor():
    frame = SyntheticSource(640, 360).read()[1]
    a, b = frame.copy(), frame.copy()
    draw_hud(a, HUDState())
    draw_hud(b, HUDState(cursor=(320, 200), hand=True))
    diff = int(np.count_nonzero(a != b))
    check("HUD draws cursor + hand chip", diff > 100, f"{diff} px differ")


def t_real_tracker():
    required = os.environ.get("AIRSOP_REQUIRE_MEDIAPIPE") == "1"
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        if required:
            check("MediaPipe Hand Landmarker loads", False, "not installed")
        else:
            print(f"SKIP  {'MediaPipe Hand Landmarker loads':<44} "
                  "mediapipe not installed (optional)")
        return
    from src.hands import HandTracker
    try:
        t = HandTracker()
        frame = SyntheticSource(640, 360).read()[1]
        obs = [t.process(frame) for _ in range(3)]
        t.close()
        check("MediaPipe Hand Landmarker loads",
              all(o is None for o in obs), "no hand in test pattern -> None")
    except Exception as e:  # noqa: BLE001
        check("MediaPipe Hand Landmarker loads", False, repr(e)[:60])


if __name__ == "__main__":
    t_synthetic_shape()
    t_synthetic_deterministic()
    t_hud_draws()
    t_hud_with_step_state()
    t_headless_loop()
    t_fps_counter()
    t_synthetic_hand()
    t_palm_anchor()
    t_ema()
    t_cursor_tracks_truth()
    t_hud_cursor()
    t_real_tracker()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks PASS")
    sys.exit(0 if all(RESULTS) else 1)
