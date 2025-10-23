# TenukiGo-Pi
## Setup Instructions

This project uses **Micromamba (or conda/mamba)** for environment management. This is the recommended way to ensure complex dependencies like OpenCV and PyTorch are installed correctly.

---

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

You are now ready to run the script.

---

## How to Run

After completing the setup steps:

1.  **Ensure your environment is active.** Your terminal prompt should show `(tenukigo_pi)`. If not, activate it.

2.  **Make sure you are in the project root directory** (`TenukiGo-Pi/`).

3.  **(Optional) Place your video file** in the `data/` directory. This keeps the project root clean.

4.  **Run the main script**, providing the path to your Go game video file.

    **Example using the test video:**
    ```bash
    python scripts/process_video.py --video data/test.mp4
    ```

    **Example using your own video:**
    ```bash
    python scripts/process_video.py --video data/my_game.mp4
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