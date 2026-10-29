# AirSOP

Touchless laboratory bench workstation: gesture-controlled digital SOPs
(Standard Operating Procedures) with vision-based labware verification and a
tamper-evident audit trail. Built for the paper-to-digital transition at the
lab bench: gloved technicians in sterile environments step through protocols
without touching a keyboard, the camera verifies the physical setup matches
each step, and every confirmation writes a timestamped, hash-chained audit
record.

Status: **M1 code complete** (hand tracking with a palm-anchored, EMA
(Exponential Moving Average) smoothed cursor; 13/13 headless checks PASS).
The live gloved-hand check is the remaining M1 exit item. See the roadmap
below.

[![ci](https://github.com/arewa-builds/airsop/actions/workflows/ci.yml/badge.svg)](https://github.com/arewa-builds/airsop/actions/workflows/ci.yml)

## Run it in one step

Docker is the only requirement. The image bundles Python, OpenCV (Open
Source Computer Vision Library), MediaPipe, the system graphics libraries,
and the hand model, so it runs the same on any machine.

```bash
docker run --rm ghcr.io/arewa-builds/airsop
```

That runs the headless selfcheck and should print `14/14 checks PASS`.

| Goal | Command |
|---|---|
| Selfcheck from a clone | `docker compose run --rm selfcheck` or `make check` |
| Render a synthetic demo to `out/demo.mp4` | `docker compose run --rm demo` or `make demo` |
| GitHub Codespaces | Open the repo in a Codespace. The dev container builds from the same Dockerfile and runs the selfcheck on create |
| Live webcam on a laptop | `make webcam` (or the native steps below) |

The webcam stays native on purpose. Docker on macOS and Windows cannot reach
the host camera or open windows, so live tracking runs in a local virtual
environment, while everything provable without hardware runs in the
container.

### Deployment rule for every milestone

Each milestone (M0 to M5) ships only when all three are green:

1. `python tests/selfcheck.py` with core dependencies on Python 3.10, 3.12
   and 3.14 (CI (Continuous Integration) matrix).
2. The same selfcheck inside the Docker image, with MediaPipe required.
3. A synthetic demo render from the image (uploaded as a CI artifact).

Pushes to `master` that pass all three publish
`ghcr.io/arewa-builds/airsop:latest` plus a commit-tagged image.

## Native quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-camera.txt   # laptop: webcam + MediaPipe
# pip install -r requirements.txt        # headless only (servers, Codespaces)

python -m src.main                  # webcam 0, windowed (ESC quits)
python -m src.main --synthetic      # no camera? use the synthetic source
python tests/selfcheck.py           # headless smoke tests, no hardware needed
```

On a minimal Linux image without Docker, the desktop OpenCV build fails
with `libGL.so.1: cannot open shared object file`. Use
`requirements.txt` (headless OpenCV), or install `libgl1 libglib2.0-0
libegl1 libgles2` from the system package manager. The hand model
(`hand_landmarker.task`, about 7.8 MB) downloads to `models/` on first
live run.

## Hand tracking (M1)

- `HandTracker` wraps the MediaPipe Hand Landmarker (Tasks API, VIDEO
  mode). The import is lazy, so the selfcheck passes without MediaPipe.
- `SyntheticHandTracker` emits 21 procedural landmarks orbiting the
  synthetic reticle, plus jitter, with the same interface. The full loop
  runs headless with known ground truth.
- The cursor anchors to the palm center (mean of the wrist and four MCP
  (metacarpophalangeal) joints), not a fingertip. A unit test proves a
  moving index tip does not move the cursor.
- `CursorEMA` with alpha 0.4 cuts jitter standard deviation from 0.0086 to
  0.0037 in the selfcheck, and the smoothed cursor tracks the moving
  ground truth within about 16 px at 1280 px width.

## The synthetic source

`SyntheticSource` generates deterministic frames with a moving reticle, so
the capture loop, HUD, and future gesture logic can be developed and tested
with ground truth known in advance, before trusting the pipeline with
hardware. Same philosophy as a flight simulator: prove the system in a world
you control first.

## Repository structure

```
airsop/
├── README.md
├── PROJECT_PLAN.md         # full end-to-end plan and pipeline
├── requirements.txt        # headless core (numpy, opencv-python-headless)
├── requirements-camera.txt # laptop: numpy + mediapipe
├── Dockerfile              # one-step, hardware-free proof
├── docker-compose.yml      # selfcheck and demo services
├── Makefile                # check, demo, native-check, webcam
├── .devcontainer/          # Codespaces builds from the Dockerfile
├── .github/workflows/ci.yml
├── protocols/              # SOP definitions as JSON config (M3)
├── src/
│   ├── main.py             # capture loop and wiring
│   ├── capture.py          # CameraSource + SyntheticSource
│   ├── hands.py            # HandTracker, SyntheticHandTracker, CursorEMA
│   └── hud.py              # overlay rendering (HUDState driven)
├── tests/
│   ├── fixtures/           # recorded landmark sequences (M2)
│   └── selfcheck.py        # headless smoke tests
└── assets/
```

## Roadmap

| Milestone | Scope | Exit | Status |
|---|---|---|---|
| M0 | Scaffold, capture loop, HUD skeleton | Overlay renders, selfcheck green | done |
| M1 | Hand tracking, EMA-smoothed cursor | Tracks with nitrile gloves | code done, glove check pending |
| M2 | Gesture classifier + hold-to-trigger debounce | Fixture replay PASS table | next |
| M3 | SOP engine + hash-chained audit log | Full protocol run by gesture | |
| M4 | Labware verification (rack, cap colors) | Step blocks on missing tube | |
| M5 | Demo video + README strategy section | 60-90 s recorded run | |

## License

MIT (see LICENSE).
