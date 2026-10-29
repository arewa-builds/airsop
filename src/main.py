"""main.py - AirSOP capture + hand loop (M0 + M1).

Usage:
    python -m src.main                 # webcam 0 + MediaPipe hands, windowed
    python -m src.main --synthetic     # synthetic frames + synthetic hand
    python -m src.main --synthetic --headless --frames 90   # smoke run
    python -m src.main --synthetic --headless --record out/demo.mp4

M1 cursor: palm-center landmark, EMA smoothed, rendered by the HUD.
ESC quits the windowed loop.
"""
from __future__ import annotations

import argparse

import cv2

from .capture import CameraSource, SyntheticSource, FpsCounter
from .hands import HandTracker, SyntheticHandTracker, CursorEMA
from .hud import HUDState, draw_hud


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
    p.add_argument("--no-mirror", action="store_true",
                   help="do not flip the webcam image horizontally")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        source = SyntheticSource(args.width, args.height)
        tracker = SyntheticHandTracker()
    else:
        source = CameraSource(args.camera, args.width, args.height)
        tracker = HandTracker(model_path=args.model)

    fps = FpsCounter()
    cursor = CursorEMA(alpha=0.4)
    state = HUDState()
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
            if obs is not None:
                p = cursor.update(obs.palm)
                state.cursor = (int(p[0] * frame.shape[1]),
                                int(p[1] * frame.shape[0]))
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
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"frames={shown}  achieved_fps={state.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
