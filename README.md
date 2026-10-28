# AirSOP

Touchless laboratory bench workstation: gesture-controlled digital SOPs
(Standard Operating Procedures) with vision-based labware verification and a
tamper-evident audit trail. Built for the paper-to-digital transition at the
lab bench: gloved technicians in sterile environments step through protocols
without touching a keyboard, the camera verifies the physical setup matches
each step, and every confirmation writes a timestamped, hash-chained audit
record.

Status: **M0 complete** (scaffold, capture loop, HUD skeleton, headless
selfcheck). See the roadmap below.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.main                  # webcam 0, windowed (ESC quits)
python -m src.main --synthetic      # no camera? use the synthetic source
python tests/selfcheck.py           # headless smoke tests, no hardware needed
```

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
├── requirements.txt
├── protocols/              # SOP definitions as JSON config (M3)
├── src/
│   ├── main.py             # capture loop and wiring
│   ├── capture.py          # CameraSource + SyntheticSource
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
| M1 | Hand tracking, EMA-smoothed cursor | Tracks with nitrile gloves | next |
| M2 | Gesture classifier + hold-to-trigger debounce | Fixture replay PASS table | |
| M3 | SOP engine + hash-chained audit log | Full protocol run by gesture | |
| M4 | Labware verification (rack, cap colors) | Step blocks on missing tube | |
| M5 | Demo video + README strategy section | 60-90 s recorded run | |

## License

MIT (see LICENSE).
