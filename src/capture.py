"""capture.py - frame sources for AirSOP.

CameraSource wraps a real webcam. SyntheticSource generates a deterministic
test pattern with a moving reticle, so the capture loop and HUD can be
developed and tested headless, with ground truth known in advance. Same
philosophy as a synthetic scene generator: test the pipeline before
trusting it with hardware.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


class FpsCounter:
    """Sliding-window FPS meter over the last `window` frames."""

    def __init__(self, window: int = 30):
        self.window = window
        self._stamps: list[float] = []

    def tick(self) -> float:
        now = time.monotonic()
        self._stamps.append(now)
        if len(self._stamps) > self.window:
            self._stamps.pop(0)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class CameraSource:
    """Real webcam source."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera index {index} did not open")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def release(self) -> None:
        self.cap.release()


class SyntheticSource:
    """Deterministic synthetic frames: moving reticle + frame counter.

    Paced to `fps` so the loop behaves like a real camera feed.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0,
                 seed: int = 7):
        self.width, self.height = width, height
        self.period = 1.0 / fps
        self.rng = np.random.default_rng(seed)
        self._n = 0
        self._next = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray]:
        self._next += self.period
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)

        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        t = self._n / 30.0
        cx = int(self.width / 2 + (self.width / 3) * np.sin(t))
        cy = int(self.height / 2 + (self.height / 4) * np.cos(0.7 * t))
        cv2.drawMarker(frame, (cx, cy), (0, 200, 200),
                       cv2.MARKER_CROSS, 40, 2, cv2.LINE_AA)
        cv2.putText(frame, f"synthetic frame {self._n}", (20, self.height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1,
                    cv2.LINE_AA)
        self._n += 1
        return True, frame

    def release(self) -> None:
        pass
