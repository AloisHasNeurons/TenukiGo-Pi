FROM python:3.10-slim-bookworm

# 1. Install ONLY minimal system libraries
# We need libglib2.0-0 for some OpenCV internal dependencies, even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest | tar -xj -C /usr/local/bin/ --strip-components=1 bin/micromamba

WORKDIR /app
COPY environment.yml /app/

# FIX FOR QEMU HANGS (Prevents freezing during build)
ENV MAMBA_EXTRACT_THREADS=1

# 3. Create Environment
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN micromamba create -y -f environment.yml && \
    micromamba clean --all --yes

# 4. Set Up Environment Variables
# We define the path explicitly so we can use it for pip installation below
ENV CONDA_ENV_PATH="/opt/conda/envs/tenukigo_pi"
ENV PATH="$CONDA_ENV_PATH/bin:$PATH"

# 5. Install Application
# CRITICAL: We use "$CONDA_ENV_PATH/bin/pip" to guarantee libraries 
# are installed into the Conda env, not the system python.
COPY lib/ /app/lib/
RUN "$CONDA_ENV_PATH/bin/pip" install --no-cache-dir /app/lib/sente-*.whl

COPY pyproject.toml README.md /app/
COPY main.py /app/
COPY src/ /app/src/

# Install the app itself
RUN "$CONDA_ENV_PATH/bin/pip" install --no-cache-dir .

COPY models/ /app/models/

CMD ["tail", "-f", "/dev/null"]