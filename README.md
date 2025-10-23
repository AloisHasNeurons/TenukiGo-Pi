# TenukiGo-Pi
## Setup Instructions

You can set up this project using either **Micromamba (or conda or mamba) (recommended)** or Python's built-in **venv**. Micromamba is generally better at handling complex dependencies like OpenCV and PyTorch/TensorFlow across different operating systems.

---

### Local Setup (Micromamba - Recommended)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AloisHasNeurons/TenukiGo-Pi.git
    cd TenukiGo-Pi
    ```

2.  **Create the Micromamba environment:**
    This command reads the `environment.yml` file and installs all necessary packages.
    ```bash
    micromamba env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    micromamba activate tenukigo_pi
    ```
    *(Your terminal prompt should now show `(tenukigo_pi)` at the beginning)*

4.  **Install the package in editable mode:**
    This links your `src/` directory so Python can find your code. Run this command from the project root directory (`TenukiGo-Pi/`).
    ```bash
    pip install -e .
    ```

You are now ready to run the script (see "How to Run" section).

---

### Local Setup (venv)

Using `venv` requires you to have **Python 3.10 or higher** installed on your system. You might also need to manually install system-level dependencies for packages like OpenCV depending on your OS.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AloisHasNeurons/TenukiGo-Pi.git
    cd TenukiGo-Pi
    ```

2.  **Create a virtual environment:**
    Run this command from the project root directory (`TenukiGo-Pi/`). This creates a `venv` folder.
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Linux/macOS:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows (Command Prompt/PowerShell):**
        ```bash
        .\venv\Scripts\activate
        ```
    *(Your terminal prompt should now show `(venv)` at the beginning)*

4.  **Install dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install the package in editable mode:**
    This links your `src/` directory.
    ```bash
    pip install -e .
    ```

You are now ready to run the script (see "How to Run" section).

---

## How to Run

After completing the setup steps (either Micromamba or venv):

1.  **Ensure your environment is active.** Your terminal prompt should show either `(tenukigo_pi)` or `(venv)`. If not, activate it using the commands from the setup section.

2.  **Make sure you are in the project root directory** (`TenukiGo-Pi/`).

3.  **Run the main script**, providing the path to your Go game video file using the `--video` argument:
    ```bash
    python scripts/process_video.py --video /path/to/your/game.mp4
    ```
    * Replace `/path/to/your/game.mp4` with the actual path to your video.

4.  **Wait for processing.** The script will load the models and process the video frame by frame (based on the `ANALYSIS_INTERVAL` set in the script). You'll see log messages in the terminal.

5.  **Check the output.** Once the script finishes, a file named `game_output.sgf` (or whatever you specify with the `--output` argument) will be created in your project root directory. This file contains the recorded game in SGF format.

### Optional Arguments

* `--yolo-model <path>`: Specify a different path for the YOLO model (`.pt`).
* `--keras-model <path>`: Specify a different path for the Keras model (`.keras`).
* `--output <path>`: Specify a different name or location for the output SGF file.

**Example with options:**
```bash
python scripts/process_video.py --video my_game.mp4 --output results/final_game.sgf