FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    rpicam-apps \
    ffmpeg \
    libcamera-0 \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY libs/ /app/libs/
RUN pip install /app/libs/sente-*.whl

COPY pyproject.toml README.md /app/
RUN pip install .

COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

RUN chmod +x /app/scripts/*.sh

CMD ["tail", "-f", "/dev/null"]