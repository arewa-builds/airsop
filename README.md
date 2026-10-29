# AirSOP

Touchless laboratory bench workstation: gesture-controlled digital SOPs
(Standard Operating Procedures) with vision-based labware verification and a
tamper-evident audit trail. Built for the paper-to-digital transition at the
lab bench: gloved technicians in sterile environments step through protocols
without touching a keyboard, the camera verifies the physical setup matches
each step, and every confirmation writes a timestamped, hash-chained audit
record.

Status: **M2 complete** (six-gesture classifier, hold-to-trigger debouncer,
fixture replay; 20/20 headless checks PASS, 21/21 in Docker). M1 hand
tracking is code complete; the live gloved-hand check is its remaining exit
item. See the roadmap below.

[![ci](https://github.com/arewa-builds/airsop/actions/workflows/ci.yml/badge.svg)](https://github.com/arewa-builds/airsop/actions/workflows/ci.yml)

## Run it in one step

Docker is the only requirement. The image bundles Python, OpenCV (Open
Source Computer Vision Library), MediaPipe, the system graphics libraries,
and the hand model, so it runs the same on any machine.

```bash
docker run --rm ghcr.io/arewa-builds/airsop
```

That runs the headless selfcheck and should print `21/21 checks PASS`.

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

## Gestures (M2)

Six gestures, each mapped to one bench action. A gesture fires only after
it is held for 18 frames (0.6 s at 30 fps), fires once, and must be
released before it fires again. Tracking dropouts of up to 3 frames do not
reset the hold, which matters with gloves.

| Gesture | Shape | AirSOP action |
|---|---|---|
| `point_right` | index only, pointing right | next step |
| `point_left` | index only, pointing left | previous step |
| `pinch` | thumb meets index tip, other three fingers up (OK sign) | confirm step |
| `fist` | all four fingers curled | lock or unlock; the only gesture honored while locked |
| `open_palm` | all fingers extended | pause |
| `peace` | index and middle up | help |

The classifier is geometric, not learned. A finger counts as extended when
its tip is more than 1.10x farther from the wrist than its PIP (proximal
interphalangeal) joint; pinch is thumb-to-index distance under 0.25 palm
lengths; a point counts as horizontal when the index direction is within
about 35 degrees of the screen axis. All tests are ratios of 2D distances,
so they hold across hand size, camera distance, in-plane rotation, and left
or right hand. Depth (z) is ignored because it is the noisiest landmark
coordinate. Pointing straight up is deliberately not a command.

Why rules instead of a model: every decision is explainable in one line
and auditable, there is no training set to collect, and thresholds can be
reviewed like any other controlled parameter.

### Fixture replay

`tests/fixtures/*.jsonl` are landmark sequences (one meta line, then one
line per frame) replayed through the same classifier and debouncer as the
live loop. The selfcheck prints a PASS table:

| Fixture | What it proves |
|---|---|
| `each_gesture_once` | all six gestures fire, in order |
| `flicker_rejected` | 10-frame gestures (under the hold) never fire |
| `dropout_tolerated` | two 2-frame tracking losses do not break a hold |
| `long_hold_fires_once` | a 4 s fist fires once, no auto-repeat |
| `release_rearms` | release, then the same gesture fires again |
| `non_command_pose_ignored` | 2 s of pointing up fires nothing |
| `protocol_walk_left_hand_tilted` | 7-step bench run with a left hand, 15 degree tilt, smaller hand, heavier jitter |

These fixtures are synthetic, built by `python tools/make_fixtures.py` from
procedural poses (`src/poses.py`) with jitter, drift, and dropouts. The
selfcheck also proves the committed files match the generator. Synthetic
poses are cleaner than real tracking, so the selfcheck adds a noise
check: with landmark jitter at about 7% of palm length, frame accuracy is
99.2% (it falls to about 91% at 10% and 59% at 17%).

Record real fixtures, gloved or bare, with the live loop:

```bash
python -m src.main --record-landmarks tests/fixtures/real_gloved_walk.jsonl \
    --expect point_right,pinch,point_left
```

Any `.jsonl` in `tests/fixtures/` is replayed automatically.

### Shared core decision

The vocabulary matches SterileView's planned six gestures exactly, so the
engine is portable. It stays inside AirSOP for now: the rule is not to pay
for abstraction before the second use case is real. When SterileView M2
consumes the same `gestures.py` unchanged, both repos switch to a shared
`airgesture` package. If the vocabularies diverge, the engines stay
separate and this section will say why.

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
│   ├── gestures.py         # classify(), HoldDebouncer, ACTIONS
│   ├── poses.py            # procedural 21-landmark hand poses
│   ├── replay.py           # fixture replay through the live pipeline
│   └── hud.py              # overlay rendering (HUDState driven)
├── tests/
│   ├── fixtures/           # landmark replay fixtures (.jsonl)
│   └── selfcheck.py        # headless smoke tests
├── tools/
│   └── make_fixtures.py    # regenerates the synthetic fixtures
└── assets/
```

## Roadmap

| Milestone | Scope | Exit | Status |
|---|---|---|---|
| M0 | Scaffold, capture loop, HUD skeleton | Overlay renders, selfcheck green | done |
| M1 | Hand tracking, EMA-smoothed cursor | Tracks with nitrile gloves | code done, glove check pending |
| M2 | Gesture classifier + hold-to-trigger debounce | Fixture replay PASS table | done |
| M3 | SOP engine + hash-chained audit log | Full protocol run by gesture | next |
| M4 | Labware verification (rack, cap colors) | Step blocks on missing tube | |
| M5 | Demo video + README strategy section | 60-90 s recorded run | |

## License

MIT (see LICENSE).
