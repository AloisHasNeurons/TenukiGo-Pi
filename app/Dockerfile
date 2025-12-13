FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gpg \
    ca-certificates \
    && \
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian/ bookworm main" | tee /etc/apt/sources.list.d/raspi.list \
    && \
    apt-get update && apt-get install -y --no-install-recommends \
    rpicam-apps \
    ffmpeg \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY lib/ /app/lib/
RUN pip install --no-cache-dir /app/lib/sente-*.whl

COPY pyproject.toml README.md /app/
COPY main.py /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir .

COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

COPY models/ /app/models/


CMD ["tail", "-f", "/dev/null"]