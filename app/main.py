"""
TenukiGo-Pi: Video to SGF Processor

Process Go game videos to generate SGF files.
"""

import argparse
import logging
import os
import cv2
import sente
from tenukigo_pi.GoGame import GoGame
from tenukigo_pi.GoBoard import GoBoard
from tenukigo_pi.GoVisual import GoVisual

from tenukigo_pi.utils.model_utils import load_corrector_model
from tenukigo_pi.corrector_noAI import corrector_no_ai
from tenukigo_pi.utils.sgf_utils import to_sgf

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constants ---
ANALYSIS_INTERVAL = 0.1  # seconds
DEFAULT_YOLO_PATH = os.path.join("models", "model.pt")
DEFAULT_KERAS_PATH = os.path.join("models", "modelCNN.keras")
DEFAULT_OUTPUT_SGF = os.path.join("outputs", "game_output.sgf")
MAX_INIT_FRAMES = 300


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process a Go game video and generate an SGF file."
    )
    parser.add_argument(
        "-v", "--video", type=str, required=True,
        help="Path to the input video file (e.g., data/test.mp4)."
    )
    parser.add_argument(
        "-y", "--yolo-model", type=str, default=DEFAULT_YOLO_PATH,
        help=f"Path to the YOLO model file (default: {DEFAULT_YOLO_PATH})."
    )
    parser.add_argument(
        "-k", "--keras-model", type=str, default=DEFAULT_KERAS_PATH,
        help=f"Path to the Keras model (default: {DEFAULT_KERAS_PATH})."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=DEFAULT_OUTPUT_SGF,
        help=f"Path to save the SGF file (default: {DEFAULT_OUTPUT_SGF})."
    )
    parser.add_argument(
        "--transparent", action="store_true", default=True,
        help="Use transparent mode (records states for AI post-processing)."
    )
    parser.add_argument(
        "--real-time", dest='transparent', action='store_false',
        help="Use real-time mode (detects moves frame-by-frame)."
    )
    return parser.parse_args()


def initialize_board(cap: cv2.VideoCapture,
                     go_game: GoGame) -> bool:
    """Try to find and initialize the board from video frames."""
    logger.info("Finding board in video...")
    frame_count_init = 0

    while cap.isOpened() and frame_count_init < MAX_INIT_FRAMES:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Video ended before board could be initialized.")
            return False

        frame_count_init += 1

        try:
            # Use end_game=False, we don't need SGF yet
            _, _ = go_game.initialize_game(frame, end_game=False)
            logger.info(
                f"Board initialized successfully on frame {frame_count_init}!"
            )
            return True
        except Exception as e:
            if frame_count_init % 30 == 0:
                logger.info(
                    f"Tried {frame_count_init} frames, still searching..."
                )
            logger.debug(f"Init frame {frame_count_init} failed: {e}")
            continue

    logger.error(
        f"Could not initialize board after {MAX_INIT_FRAMES} frames."
    )
    return False


def process_video(cap: cv2.VideoCapture, go_game: GoGame) -> int:
    """Process the video frame-by-frame after initialization."""
    logger.info("Processing video to detect moves...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.warning("Video FPS is 0. Defaulting to 30.")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * ANALYSIS_INTERVAL))

    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}, "
                f"Analyzing every {frame_interval} frames")

    processed_frames = 1
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video file reached.")
            break

        frame_count += 1

        if frame_count % frame_interval != 0:
            continue

        processed_frames += 1

        try:
            if processed_frames % 10 == 0:
                logger.info(f"Processed {processed_frames} analysis frames... "
                            f"(video frame {frame_count}/{total_frames})")

            _, _ = go_game.main_loop(frame, end_game=False)

        except Exception as e:
            # Log warnings less frequently to avoid spam
            if processed_frames % 20 == 0:
                logger.warning(f"Error processing frame {frame_count}: {e}")
            logger.debug(f"Full error on frame {frame_count}: {e}",
                         exc_info=True)
            continue

    logger.info(f"Processing complete. Analyzed {processed_frames} frames.")
    return processed_frames


def run_pipeline(args: argparse.Namespace):
    """Initialize and run the full video processing pipeline."""
    logger.info(f"Loading YOLO model from: {args.yolo_model}")
    go_board = GoBoard(model_path=args.yolo_model)

    logger.info(f"Loading Keras corrector model from: {args.keras_model}")
    corrector_model = load_corrector_model(model_path=args.keras_model)

    logger.info("Initializing GoGame engine...")
    game = sente.Game()
    go_visual = GoVisual(game)
    use_transparent = args.transparent

    go_game = GoGame(
        game=game,
        board_detect=go_board,
        go_visual=go_visual,
        corrector_model=corrector_model,
        transparent_mode=use_transparent
    )

    mode = "TRANSPARENT (AI Post-Processing)" if use_transparent else (
        "REAL-TIME"
    )
    logger.info(f"Running in {mode} mode")

    logger.info(f"Opening video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Could not open video file {args.video}")
        return

    # --- 1. Initialize Board ---
    if not initialize_board(cap, go_game):
        cap.release()
        return

    # --- 2. Process Video ---
    processed_frames = process_video(cap, go_game)
    cap.release()
    cv2.destroyAllWindows()

    # --- 3. Post-Process and Save SGF ---
    final_sgf = None
    if use_transparent:
        num_states = len(go_game.numpy_board)
        logger.info(f"Running AI post-processing on {num_states} "
                    "board states...")
        if num_states < 2:
            logger.error("Not enough board states captured for AI processing.")
        else:
            try:
                final_sgf = go_game.post_treatment(end_game=True)
                if final_sgf:
                    logger.info(
                        f"Generated SGF with {len(final_sgf)} characters"
                    )
                else:
                    logger.warning("Empty SGF generated by AI.")
            except Exception as e:
                logger.error(f"Error during AI post-processing: {e}",
                             exc_info=True)
                logger.info("Attempting fallback SGF generation (no AI)...")
                try:
                    move_list = corrector_no_ai(go_game.numpy_board)
                    final_sgf = to_sgf(move_list)
                    logger.info("Fallback SGF generation successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    final_sgf = None
    else:
        # In real-time mode, just get the SGF from the sente game object
        final_sgf = go_game.get_sgf()

    # --- 4. Save File ---
    if final_sgf:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(args.output, "w") as f:
                f.write(final_sgf)
            logger.info(f"\n✓ Successfully saved game to {args.output}")
            logger.info(f"  Total frames analyzed: {processed_frames}")
        except IOError as e:
            logger.error(f"\n✗ Error: "
                         f"Could not write SGF to {args.output}: {e}")
    else:
        logger.error("\n✗ Error: No SGF data was generated.")


if __name__ == "__main__":
    arguments = parse_arguments()
    run_pipeline(arguments)
