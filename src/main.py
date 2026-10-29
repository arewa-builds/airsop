"""main.py - AirSOP capture, hand, and gesture loop (M0 to M2).

Usage:
    python -m src.main                 # webcam 0 + MediaPipe hands, windowed
    python -m src.main --synthetic     # synthetic frames + synthetic hand
    python -m src.main --synthetic --headless --frames 90   # smoke run
    python -m src.main --synthetic --headless --record out/demo.mp4

    python -m src.main --record-landmarks tests/fixtures/real_pinch.jsonl \
        --expect pinch                 # capture a real (gloved) fixture

M1 cursor: palm-center landmark, EMA smoothed, rendered by the HUD.
M2 gestures: classify() per frame, HoldDebouncer fires once per hold, and
the event maps to an AirSOP action (gestures.ACTIONS). Fist toggles the
lock; while locked, only fist is honored. ESC quits the windowed loop.
"""
from __future__ import annotations

import argparse

import cv2

from .capture import CameraSource, SyntheticSource, FpsCounter
import json

from .gestures import ACTIONS, HoldDebouncer, NONE, classify
from .hands import HandTracker, SyntheticHandTracker, CursorEMA
from .hud import HUDState, draw_hud


# Synthetic demo: a short bench run, then lock and unlock.
DEMO_SCRIPT = [(None, 15), ("point_right", 25), (None, 8), ("pinch", 25),
               (None, 8), ("point_right", 25), (None, 8), ("pinch", 25),
               (None, 8), ("point_left", 25), (None, 8), ("peace", 25),
               (None, 8), ("fist", 25), (None, 8), ("pinch", 25), (None, 8),
               ("fist", 25), (None, 8), ("open_palm", 25)]


def apply_event(state, gesture: str) -> str | None:
    """Update HUD state for a fired gesture. Returns the action taken."""
    action = ACTIONS[gesture]
    if action == "lock":
        state.locked = not state.locked
        action = "lock" if state.locked else "unlock"
    elif state.locked:
        state.last_action = f"{gesture} ignored (locked)"
        return None
    state.last_action = f"{gesture} -> {action}"
    return action


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="airsop")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--synthetic", action="store_true",
                   help="deterministic synthetic frames and hand")
    p.add_argument("--headless", action="store_true",
                   help="no window; run --frames frames and exit")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", default=None,
                   help="path to hand_landmarker.task (auto-downloaded)")
    p.add_argument("--record", default=None, metavar="PATH",
                   help="also write the HUD output to an .mp4 file")
    p.add_argument("--hold", type=int, default=18,
                   help="frames a gesture must be held to fire (18 = 0.6 s)")
    p.add_argument("--record-landmarks", default=None, metavar="PATH",
                   help="write per-frame landmarks as a replay fixture (.jsonl)")
    p.add_argument("--expect", default="",
                   help="comma-separated expected events for the fixture meta")
    p.add_argument("--no-mirror", action="store_true",
                   help="do not flip the webcam image horizontally")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        source = SyntheticSource(args.width, args.height)
        tracker = SyntheticHandTracker(script=DEMO_SCRIPT)
    else:
        source = CameraSource(args.camera, args.width, args.height)
        tracker = HandTracker(model_path=args.model)

    fps = FpsCounter()
    cursor = CursorEMA(alpha=0.4)
    state = HUDState()
    debouncer = HoldDebouncer(hold_frames=args.hold)
    rec = open(args.record_landmarks, "w") if args.record_landmarks else None
    if rec:
        expected = [e for e in args.expect.split(",") if e]
        rec.write(json.dumps({"type": "meta", "name": args.record_landmarks,
                              "source": "synthetic" if args.synthetic
                              else "camera", "fps": 30,
                              "expected_events": expected}) + "\n")
    shown = 0
    writer = None
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            if not args.synthetic and not args.no_mirror:
                frame = cv2.flip(frame, 1)       # mirror: hand moves with you
            obs = tracker.process(frame)
            state.hand = obs is not None
            state.skeleton = None if obs is None else [
                (int(x * frame.shape[1]), int(y * frame.shape[0]))
                for x, y in obs.landmarks[:, :2]]
            if obs is not None:
                p = cursor.update(obs.palm)
                state.cursor = (int(p[0] * frame.shape[1]),
                                int(p[1] * frame.shape[0]))
            label = classify(obs)
            state.gesture = label if label != NONE else "--"
            fired = debouncer.update(label)
            state.hold = debouncer.progress
            if fired:
                apply_event(state, fired)
                print(f"event frame={shown} {state.last_action}")
            if rec:
                rec.write(json.dumps({
                    "type": "frame", "i": shown, "label": "unlabeled",
                    "landmarks": None if obs is None else
                    [[round(float(v), 4) for v in row]
                     for row in obs.landmarks]},
                    separators=(",", ":")) + "\n")
            state.fps = fps.tick()
            draw_hud(frame, state)
            shown += 1
            if args.record:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (w, h))
                writer.write(frame)

            if args.headless:
                if shown >= args.frames:
                    break
            else:
                cv2.imshow("AirSOP", frame)
                if cv2.waitKey(1) & 0xFF == 27:      # ESC
                    break
    finally:
        source.release()
        tracker.close()
        if writer is not None:
            writer.release()
        if rec:
            rec.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"frames={shown}  achieved_fps={state.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
