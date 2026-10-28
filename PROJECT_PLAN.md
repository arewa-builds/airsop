# AirSOP: Touchless Laboratory Bench Workstation

End-to-end plan and pipeline for a computer-vision interaction layer that lets
gloved technicians step through digital SOPs (Standard Operating Procedures)
hands-free, with vision-based labware verification and a tamper-evident audit
trail. Built as public portfolio proof for laboratory automation and digital
enablement leadership.

---

## Why this project exists

Laboratory technicians in biosafety cabinets and cleanrooms cannot touch
keyboards without breaking sterility, yet most bench protocols still live in
paper binders or on a monitor across the room. AirSOP demonstrates the
paper-to-digital transition for one bench: gestures step through the protocol,
a camera verifies the physical setup matches each step's expectation, and
every confirmation writes a timestamped, hash-chained audit record.

Career alignment: this is concrete proof for laboratory operations and
automation leadership conversations. It extends the
PharmaVerse line-clearance story from synthetic environments to a live bench,
and it reuses the perception challenge DNA (zones become regions of interest,
the cluster rule becomes a detection quorum, the debouncer becomes gesture
hysteresis).

## Success definition

- Public GitHub repo that runs end to end on a laptop with a webcam.
- A 60 to 90 second screen-recorded demo video (the LinkedIn artifact).
- A README with a strategy section written at director level: pilot plan,
  metrics, integration path, validation notes.

## System pipeline

```
webcam
  └─► capture loop (OpenCV, 30 fps target)
        ├─► hand tracking (MediaPipe Hands, 21 landmarks per hand)
        │     └─► gesture classifier (geometric rules on landmarks)
        │           └─► debouncer (hold-to-trigger state machine)
        │                 └─► gesture events (CONFIRM, NEXT, PREV, LOCK, ZOOM)
        ├─► labware check (fixed rack region of interest)
        │     └─► verification result (tube count, cap colors vs expected)
        ▼
  SOP engine (protocol defined in JSON, not code)
        ├─► HUD overlay (cursor, step banner, detections, confidence)
        └─► audit log (append-only JSONL, hash-chained records)
```

Data flow per frame: frame in, landmarks out, gesture candidate classified,
candidate must hold N frames to fire, event advances the SOP engine if the
current step's labware check passes, engine updates the HUD and appends the
audit record. One direction, no hidden state, every stage testable alone.

## Gesture vocabulary

Deliberately small and geometrically distinct, so rules on landmark distances
and angles classify them without any trained model. Explainable beats clever
in a compliance-adjacent setting.

| Gesture | Mapping | Trigger |
|---|---|---|
| Open palm | Cursor hover | Continuous (smoothed) |
| Pinch hold | CONFIRM current step | Hold 5 frames |
| Swipe left or right | NEXT or PREV step | Velocity plus hold |
| Fist | LOCK or unlock interface | Hold 8 frames |
| Two-hand spread | ZOOM plate map or image | Continuous |

Cursor smoothing uses an EMA (Exponential Moving Average) so the pointer does
not jitter; the same stabilization philosophy as the robot debouncer, applied
to pixels instead of commands.

## Labware detection (MVP scope)

One rack type only: a 6 by 4 tube rack with colored caps. Fixed camera,
fixed rack region of interest, color-threshold cap detection in HSV (Hue,
Saturation, Value) space, position grid snapping. Each SOP step declares its
expectation (for example, step 3 requires 4 tubes with red caps in row A),
and the step cannot be confirmed until the check passes. YOLO (You Only Look
Once)-based general labware detection is an explicit stretch goal, not MVP.

## Audit trail and data integrity

Every CONFIRM and deviation flag appends a JSONL (JSON Lines) record:
timestamp, session id, step id, gesture, verification result, and a SHA-256
hash of the previous record, forming a tamper-evident chain. This is the
electronic-records thinking behind 21 CFR Part 11 (Title 21, Code of Federal
Regulations, Part 11), implemented cheaply, and it is the detail that moves
the project from demo to compliance-adjacent tool in conversation.

## Repository structure

```
airsop/
├── README.md               # story, demo gif, strategy section
├── requirements.txt        # opencv-python, mediapipe, numpy
├── protocols/
│   └── sample_accessioning.json
├── src/
│   ├── main.py             # capture loop and wiring
│   ├── hands.py            # MediaPipe wrapper, landmark extraction
│   ├── gestures.py         # rule-based classifier
│   ├── debouncer.py        # hold-to-trigger state machine
│   ├── labware.py          # rack and cap verification
│   ├── sop.py              # protocol engine, JSON driven
│   ├── audit.py            # hash-chained JSONL logger
│   └── hud.py              # overlay rendering
├── tests/
│   ├── fixtures/           # recorded landmark sequences per gesture
│   └── selfcheck.py        # fixture replay, printed PASS table
└── assets/
    └── demo_script.md
```

## Milestones and exit criteria

### M0: scaffold and capture (day 1)
Repo, virtual environment, camera loop at stable frame rate, HUD skeleton
rendering static elements. Exit: main.py shows webcam feed with overlay frame.

### M1: hands and cursor (days 2 to 3)
Landmarks streaming, cursor follows open palm with EMA smoothing. Exit:
cursor tracks smoothly with the technician wearing blue nitrile gloves, not
just bare hands. Gloved tracking is a real risk; test it this early.

### M2: gestures and debounce (days 4 to 6)
Rule-based classifier for the five gestures, hold-to-trigger state machine,
recorded fixtures for each gesture, selfcheck replaying fixtures to a PASS
table. Exit: no camera needed to verify gesture logic; fixtures are ground
truth, the same test philosophy as the synthetic depth generator.

### M3: SOP engine and audit (days 7 to 9)
JSON protocol drives the stepper, CONFIRM advances only when valid, audit
log appends hash-chained records. Exit: run the full synthetic protocol by
gesture alone; the log replays the session exactly.

### M4: labware verification (days 10 to 12)
Rack detection and cap classification wired into step requirements. Exit:
step 3 blocks confirmation when a tube is missing or a cap color is wrong,
and the HUD says why.

### M5: polish and launch (days 13 to 15)
Demo script rehearsed twice, 60 to 90 second video recorded, README strategy
section written, LinkedIn post drafted. Exit: video shows one full protocol
run including one deliberate deviation caught by the system.

Roughly three weeks part-time alongside the job search.

## Testing philosophy

Fixtures over cameras. Record landmark sequences once per gesture, replay
them against the classifier deterministically, print a PASS table exactly
like the perception challenge selfcheck. Camera-in-the-loop testing happens
only at milestone exits. Deterministic inputs, ground truth known in advance,
failures reproducible.

## Strategy section content (for the README)

Written for the operations-leadership lens, not the engineer lens:

- **Pilot plan:** one bench, one assay protocol, four weeks, baseline
  measured first.
- **Metrics:** deviations caught per run, time per run versus paper baseline,
  paper steps eliminated, audit-record completeness.
- **Integration path:** audit log maps to LIMS (Laboratory Information
  Management System) fields; the SOP JSON mirrors how protocol authors would
  configure new assays without code changes.
- **Validation note:** what GxP (Good Practice) deployment would require:
  access controls, validation protocols, environmental hardening. Honest
  about the gap between demo and regulated deployment.
- **ROI framing:** ROI (Return on Investment) in minutes saved per run and
  deviations prevented, not model accuracy.

## Risks and mitigations

- **Gloved hand tracking degrades.** Test gloves in M1, not M5. Fallback:
  wrist-keypoint cursor mode with coarser gestures.
- **Lighting variability in the demo space.** Fixed camera position and a
  consistent background for the rack region; record the demo in the same
  setup used for development.
- **Scope creep into general instrument detection.** YOLO is a stretch goal
  only; one rack type done well beats five done poorly.
- **Privacy optics.** Frames are processed in memory, never stored; audit
  snapshots cover the bench region only, never faces.

## Resume and LinkedIn integration

Resume line for the medtech track: "Built a touchless bench-workflow
assistant (Python, OpenCV, MediaPipe) that steps gloved technicians through
digital SOPs with gesture confirmation and tamper-evident audit records,
targeting paper-to-digital enablement in laboratory operations."

LinkedIn post mirrors the YOLOvX format that inspired this: hook line, short
demo video, repo link. Draft hook: "Paper SOPs are the last analog island in
the modern lab. I built a touchless bench station that steps gloved hands
through a protocol and writes its own audit trail."

## Interview sound bites

- "The gesture stabilizer is the same hysteresis as my robot debouncer:
  escalate fast on strong evidence, recover slowly to prevent flapping."
- "Protocols are JSON config, not code, because operations teams need to
  author new assays without a developer. That is the digital enablement
  lesson in miniature."
- "The audit chain is tamper-evident by construction, which is the cheapest
  possible nod to Part 11 thinking, and it changes how reviewers read the
  whole project."
