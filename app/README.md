# TenukiGo Application

This directory contains the source code for the core TenukiGo application. It is designed to run in two environments:
1.  **Embedded (Raspberry Pi)**: via Docker + TensorFlow Lite Runtime.
2.  **Development (PC/Mac)**: via Python + TensorFlow/Keras.

## Structure

*   `Dockerfile`: Production image definition (Debian Bookworm / Python 3.10).
*   `main.py`: Main entry point. Pipelines video input -> YOLO Detection -> Stone Classification -> SGF Output.
*   `src/`: The `tenukigo_pi` python package containing the Computer Vision logic.
*   `models/`: Pre-trained models:
    *   `model.pt`: YOLOv8 model for board detection.
    *   `modelCNN.tflite` / `.keras`: CNN for stone classification.
*   `lib/`: Pre-compiled wheels (e.g., `sente` C++ bindings).

---

## Local Development

You can run the analysis pipeline locally without Docker for debugging or model training.

### 1. Environment Setup
We recommend using **Micromamba** (or Conda) to manage dependencies.

```bash
# 1. Create environment from file
micromamba env create -f environment.yml

# 2. Activate environment
micromamba activate tenukigo_pi

# 3. Install the package in editable mode
pip install -e .
```

### 2. Running Analysis
```bash
python main.py \
  --video data/sample_game.mp4 \
  --output result.sgf \
  --yolo-model models/model.pt \
  --keras-model models/modelCNN.keras
```
> **Note**: The script automatically detects the runtime environment. On a PC, it uses the `.keras` model with full TensorFlow. On the Pi, it defaults to the `.tflite` model with the lightweight runtime.

---

## Docker (Production Build)

The Docker image is optimized for `linux/arm64`.

### Building the Image
When moving to production or testing on the Pi, rebuild the image:

```bash
podman build --platform linux/arm64 -t tenukigo-app:latest .
```

This image is then loaded onto the Raspberry Pi via the Ansible deployment pipeline.

### Integration with Infrastructure
In production, this container is orchestrated by `scripts/record_game.sh` (deployed on the host). The workflow is:
1.  Host script captures video using `rpicam-vid`.
2.  Host script mounts the video directory into the container.
3.  Host script triggers `docker exec ... python3 /app/main.py`.