FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    ca-certificates \
    gpg \
    && \
    curl -fsSL https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | \
    gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg \
    && \
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian/ bookworm main" | \
    tee /etc/apt/sources.list.d/raspi.list \
    && \
    apt-get update && apt-get install -y --no-install-recommends \
    rpicam-apps \
    ffmpeg \
    v4l-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C /usr/local/bin/ --strip-components=1 bin/micromamba

WORKDIR /app
COPY environment.yml /app/

# FIX FOR QEMU HANGS: Force single-threaded extraction
ENV MAMBA_EXTRACT_THREADS=1

# Create the environment 'tenukigo_pi'
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN micromamba create -y -f environment.yml && \
    micromamba clean --all --yes
ENV PATH="/opt/conda/envs/tenukigo_pi/bin:$PATH"

COPY lib/ /app/lib/
RUN pip install --no-cache-dir /app/lib/sente-*.whl

COPY pyproject.toml README.md /app/
COPY main.py /app/
COPY src/ /app/src/

RUN pip install --no-cache-dir .

COPY models/ /app/models/

CMD ["tail", "-f", "/dev/null"]