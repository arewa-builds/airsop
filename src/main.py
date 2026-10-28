"""main.py - AirSOP capture loop (M0).

Usage:
    python -m src.main                 # webcam 0, windowed
    python -m src.main --synthetic     # synthetic source, windowed
    python -m src.main --synthetic --headless --frames 90   # smoke run

ESC quits the windowed loop.
"""
from __future__ import annotations

import argparse

import cv2

from .capture import CameraSource, SyntheticSource, FpsCounter
from .hud import HUDState, draw_hud


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="airsop")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--synthetic", action="store_true",
                   help="use the deterministic synthetic source")
    p.add_argument("--headless", action="store_true",
                   help="no window; run --frames frames and exit")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        source = SyntheticSource(args.width, args.height)
    else:
        source = CameraSource(args.camera, args.width, args.height)

    fps = FpsCounter()
    state = HUDState()
    shown = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            state.fps = fps.tick()
            draw_hud(frame, state)
            shown += 1

            if args.headless:
                if shown >= args.frames:
                    break
            else:
                cv2.imshow("AirSOP", frame)
                if cv2.waitKey(1) & 0xFF == 27:      # ESC
                    break
    finally:
        source.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"frames={shown}  achieved_fps={state.fps:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
