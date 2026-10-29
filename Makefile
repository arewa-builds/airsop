# One-step targets. Docker targets need only Docker; native targets need Python 3.10+.
IMAGE ?= airsop:local
PY    ?= python3

.PHONY: check demo image native-check webcam clean

check: image                 ## headless selfcheck inside Docker
	docker run --rm $(IMAGE)

demo: image                  ## render out/demo.mp4 from synthetic input
	mkdir -p out
	docker run --rm -v "$(CURDIR)/out:/app/out" $(IMAGE) \
	  python -m src.main --synthetic --headless --frames 420 --record out/demo.mp4

image:
	docker build -t $(IMAGE) .

.venv:
	$(PY) -m venv .venv
	.venv/bin/pip install -q -r requirements-camera.txt

native-check: .venv          ## selfcheck on this machine (no Docker)
	.venv/bin/python tests/selfcheck.py

webcam: .venv                ## live webcam + hand tracking (laptop only)
	.venv/bin/python -m src.main

clean:
	rm -rf .venv out models
