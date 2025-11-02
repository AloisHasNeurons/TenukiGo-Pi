# TenukiGo-Pi

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Nix Flakes](https://img.shields.io/badge/nix-flakes-blue.svg)](https://nixos.org/)

> Automated Go game recording and analysis from video using computer vision and deep learning

---

## Features

-  **Video Processing**: Automatically process Go game videos from start to finish
-  **Board Detection**: Robust YOLO-based board and stone detection
-  **Deep Learning Classification**: Keras model for accurate stone color identification
-  **SGF Export**: Generate standard SGF files compatible with all Go software
-  **Flexible Configuration**: Customizable model paths and output locations
-  **NixOS Support**: Reproducible development environment with Nix flakes

---

## Quick Start

### Prerequisites

Choose one of the following:
- **Micromamba/Conda** (recommended for most users)
- **NixOS** with flakes enabled (for reproducible environments)

---

## Setup Instructions

### Local Setup (Micromamba / Conda)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AloisHasNeurons/TenukiGo-Pi.git
    cd TenukiGo-Pi
    ```

2.  **Create the Micromamba/Conda environment:**
    This command reads the `environment.yml` file and installs all necessary packages.
    ```bash
    # Using Micromamba (fastest)
    micromamba env create -f environment.yml

    # Or using full Conda
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    micromamba activate tenukigo_pi
    # Or
    conda activate tenukigo_pi
    ```
    *(Your terminal prompt should now show `(tenukigo_pi)` at the beginning)*

4.  **Install the package in editable mode:**
    This links your `src/` directory so Python can find your code. Run this command from the project root directory (`TenukiGo-Pi/`).
    ```bash
    pip install -e .
    ```

You are now ready to run the script!

--- 

### NixOS Setup

> [!TIP]
> Nix is a package manager that guarantees reproducible builds. It ensures everyone gets the exact same dependencies and environment, making "it works on my machine" problems disappear.

<details>
<summary><b>Click to expand installation instructions</b></summary>

This project uses Nix flakes with uv2nix for reproducible Python environments.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AloisHasNeurons/TenukiGo-Pi.git
   cd TenukiGo-Pi
   ```

2. **Enter the development environment:**
   ```bash
   nix develop
   ```

   This will:
   - Create a virtual environment in `.venv/` with all dependencies from `uv.lock`
   - Clone and patch the `sente` library with necessary fixes
   - Provide all required build tools and Python packages

3. **The virtual environment is automatically activated.** You can now run scripts directly:
   ```bash
   python scripts/process_video.py --video data/test.mp4 --output outputs/final_game.sgf
   ```
   
</details>

> [!NOTE]
> The Nix setup uses Python 3.11 and includes special handling for PyTorch/CUDA packages. The `sente` library is automatically patched and installed from source during shell initialization.

---

## How to Run

After completing the setup steps:

1.  **Ensure your environment is active.** Your terminal prompt should show `(tenukigo_pi)` for Conda/Micromamba, or `(.venv)` for Nix. If not, activate it.

2.  **Make sure you are in the project root directory** (`TenukiGo-Pi/`).

3.  **(Optional) Place your video file** in the `data/` directory. This keeps the project root clean.

4.  **Run the main script**, providing the path to your Go game video file.

    **Example using a test video:**
    ```bash
    python scripts/process_video.py --video data/test.mp4
    ```

5.  **Wait for processing.** The script will load the models and process the video. You'll see log messages in the terminal.

6.  **Check the output.** Once the script finishes, a file named `game_output.sgf` will be created in the `outputs/` directory. This file contains the recorded game in SGF format.

### Optional Arguments

* `--yolo-model <path>`: Specify a different path for the YOLO model (`.pt`).
* `--keras-model <path>`: Specify a different path for the Keras model (`.keras`).
* `--output <path>`: Specify a different name or location for the output SGF file.

**Example with options:**
```bash
python scripts/process_video.py --video data/my_game.mp4 --output outputs/final_game.sgf
```

---

## Project Structure

```
TenukiGo-Pi/
├── data/              # Input video files
├── outputs/           # Generated SGF files
├── scripts/           # Main processing scripts
├── src/               # Source code modules
├── models/            # Trained YOLO and Keras models
├── environment.yml    # Conda environment specification
├── flake.nix          # Nix flake for reproducible builds
└── uv.lock            # Locked Python dependencies
```
