# AirSOP: one-step, hardware-free proof of the pipeline.
#   docker run --rm ghcr.io/arewa-builds/airsop        -> runs the selfcheck
# Every milestone (M0..M5) must keep this image green; CI enforces it.
FROM python:3.12-slim

# System libraries MediaPipe and OpenCV need on a minimal Linux image
# (the cause of "libGL.so.1: cannot open shared object file").
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 libgles2 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRSOP_HAND_MODEL=/opt/airsop/models/hand_landmarker.task \
    AIRSOP_REQUIRE_MEDIAPIPE=1

WORKDIR /app
COPY requirements.txt requirements-camera.txt ./
RUN pip install -r requirements-camera.txt

COPY . .
# Bake the hand model into the image so containers run offline.
RUN python -c "from src.hands import resolve_model; print(resolve_model())"

RUN useradd --create-home airsop && mkdir -p /app/out && chown airsop /app/out
USER airsop

CMD ["python", "tests/selfcheck.py"]
