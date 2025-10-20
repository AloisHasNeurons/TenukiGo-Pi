"""
TenukiGo-Pi: Video to SGF Processor

This script loads a video file, analyzes it frame-by-frame,
and generates an SGF file of the detected Go game.

Usage:
    python scripts/process_video.py --video path/to/your/game.mp4
"""

import cv2
import time
import argparse
import sente
import numpy as np
import os

# Import the refactored classes from your 'src' package
from tenukigo_pi.GoGame import GoGame
from tenukigo_pi.GoBoard import GoBoard
from tenukigo_pi.GoVisual import GoVisual
from tenukigo_pi.Fill_gaps_model import load_corrector_model

# --- Constants ---
# How often to run the full analysis (in seconds)
ANALYSIS_INTERVAL = 0.5

# --- Model Paths (relative to the project root) ---
# Assumes you run this from the project root: `python scripts/process_video.py ...`
DEFAULT_YOLO_PATH = os.path.join("models", "model.pt")
DEFAULT_KERAS_PATH = os.path.join("models", "modelCNN.keras")
DEFAULT_OUTPUT_SGF = "game_output.sgf"


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a Go game video and generate an SGF file."
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    parser.add_argument(
        "-y", "--yolo-model",
        type=str,
        default=DEFAULT_YOLO_PATH,
        help="Path to the YOLO model file."
    )
    parser.add_argument(
        "-k", "--keras-model",
        type=str,
        default=DEFAULT_KERAS_PATH,
        help="Path to the Keras corrector model file."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT_SGF,
        help="Path to save the final SGF file."
    )
    return parser.parse_args()


def run_pipeline(args):
    """Initializes and runs the full video processing pipeline."""

    # --- 1. Load AI Models ---
    print(f"Loading YOLO model from: {args.yolo_model}")
    # We pass the *path* to GoBoard, which will load it.
    go_board = GoBoard(model_path=args.yolo_model)

    print(f"Loading Keras corrector model from: {args.keras_model}")
    # We load this model and pass the *object* to GoGame.
    corrector_model = load_corrector_model(model_path=args.keras_model)

    # --- 2. Initialize Game Objects ---
    print("Initializing GoGame engine...")
    game = sente.Game()
    go_visual = GoVisual(game)
    go_game = GoGame(
        game=game,
        board_detect=go_board,
        go_visual=go_visual,
        corrector_model=corrector_model,
        transparent_mode=False  # Use real-time game logic
    )

    # --- 3. Open Video File ---
    print(f"Opening video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    # --- 4. Initialization Frame ---
    # We must run `initialize_game` on the first valid frame
    # to find the board and set the initial state.
    initialized = False
    print("Finding board in video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended before board could be initialized.")
            cap.release()
            return

        try:
            # Try to initialize the game with this frame
            game_plot, sgf_text = go_game.initialize_game(
                frame,
                end_game=True  # Use 'end_game' flag for post-processing logic
            )
            initialized = True
            print("Board initialized successfully!")
            break  # Success!
        except Exception as e:
            # This frame failed (e.g., blurry, no board), try next one
            print(f"Init failed on frame: {e}")
            pass

    # --- 5. Main Processing Loop ---
    if initialized:
        print("Processing video to find moves...")
        last_check_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Video is over

            current_time = time.time()
            if (current_time - last_check_time) < ANALYSIS_INTERVAL:
                # Not time to check yet, skip this frame
                continue

            # --- It's time to run the analysis! ---
            last_check_time = current_time
            try:
                print(f"Analyzing frame at timestamp {int(current_time)}...")
                # Run the main loop logic on this frame
                game_plot, sgf_text = go_game.main_loop(
                    frame,
                    end_game=True
                )

                # Optional: Show a debug window
                # cv2.imshow("Processed Board", game_plot)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            except Exception as e:
                # Frame failed (hand in way, blur, etc.).
                # We just ignore it and try again in 2 seconds.
                print(f"Warning: Could not process frame: {e}")
                pass

    # --- 6. Cleanup & Save SGF ---
    print("Processing complete.")
    cap.release()
    cv2.destroyAllWindows()

    final_sgf = go_game.get_sgf()
    if final_sgf:
        with open(args.output, "w") as f:
            f.write(final_sgf)
        print(f"Successfully saved game to {args.output}")
    else:
        print("Error: No SGF data was generated.")


if __name__ == "__main__":
    arguments = parse_arguments()
    run_pipeline(arguments)
