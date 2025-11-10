"""
TenukiGo-Pi: Video to SGF Processor with Motion Detection

Process Go game videos using motion detection to trigger analysis.
"""

import argparse
import logging
import os
import cv2
import numpy as np
import sente
from tenukigo_pi.GoGame import GoGame
from tenukigo_pi.GoBoard import GoBoard
from tenukigo_pi.GoVisual import GoVisual

from tenukigo_pi.utils.model_utils import load_corrector_model
from tenukigo_pi.corrector_noAI import corrector_no_ai
from tenukigo_pi.utils.sgf_utils import to_sgf

# Import the motion detector (add this file to your project)
from tenukigo_pi.motion_detector import MotionDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
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
    parser.add_argument(
        "--use-motion-detection", action="store_true", default=True,
        help="Use motion detection to trigger analysis (recommended)."
    )
    parser.add_argument(
        "--fixed-interval", action="store_true",
        help="Use fixed interval analysis instead of motion detection."
    )
    parser.add_argument(
        "--analysis-interval", type=float, default=0.1,
        help="Analysis interval in seconds (for fixed-interval mode)."
    )
    parser.add_argument(
        "--min-analysis-interval", type=float, default=0.3,
        help="Minimum time between analyses in motion detection mode."
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


def process_video_motion_detection(cap: cv2.VideoCapture,
                                   go_game: GoGame,
                                   min_interval: float) -> int:
    """Process video using motion detection + board change verification."""
    motion_detector = MotionDetector(
        threshold=35,
        min_changed_pixels=600,
        stability_frames=15
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Using motion detection + board change verification")
    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")

    processed_frames = 0
    frame_count = 0
    last_analysis_time = 0
    board_region_set = False
    last_board_state = None  # Track full board state, not just count

    # Force initial state capture
    initial_capture_done = False
    initial_capture_frame = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video file reached.")
            break

        frame_count += 1
        current_time = frame_count / fps

        # Set board region once
        if not board_region_set:
            try:
                go_game.board_detect.process_frame(frame)
                corners = go_game.board_detect.results[0].boxes.xyxy[
                    go_game.board_detect.results[0].boxes.cls == 1
                ].cpu().numpy()

                if len(corners) > 0:
                    x1, y1, x2, y2 = corners[0]
                    corner_points = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ])
                    motion_detector.set_board_region(corner_points)
                    board_region_set = True
            except:
                pass

        # Force initial capture
        if not initial_capture_done and frame_count >= initial_capture_frame:
            try:
                go_game.board_detect.process_frame(frame)
                current_board = go_game.board_detect.state_to_array()
                stone_count = np.sum(current_board > 0)

                if stone_count > 0:
                    logger.info(f"Initial capture at frame {frame_count}: "
                              f"{stone_count} stones present")
                    _, _ = go_game.main_loop(frame, end_game=False)
                    last_board_state = current_board.copy()
                    last_analysis_time = current_time
                    processed_frames += 1

                initial_capture_done = True
            except Exception as e:
                logger.debug(f"Initial capture failed: {e}")

        # Detect motion
        should_analyze, motion_now = motion_detector.detect_motion(frame)

        # Analyze if conditions met
        time_since_last = current_time - last_analysis_time

        if should_analyze and time_since_last >= min_interval:
            try:
                go_game.board_detect.process_frame(frame)
                current_board = go_game.board_detect.state_to_array()

                if last_board_state is None:
                    # First detection
                    stone_count = np.sum(current_board > 0)
                    if stone_count > 0:
                        processed_frames += 1
                        logger.info(f"First board state: {stone_count} stones "
                                  f"(frame {frame_count})")
                        _, _ = go_game.main_loop(frame, end_game=False)
                        last_board_state = current_board.copy()
                        last_analysis_time = current_time
                else:
                    # Calculate difference
                    diff = current_board - last_board_state
                    stones_added = np.sum(diff > 0)
                    stones_removed = np.sum(diff < 0)

                    # Only process if net stones added (ignore pure removals)
                    if stones_added > 0 and stones_added >= stones_removed:
                        current_count = np.sum(current_board > 0)
                        last_count = np.sum(last_board_state > 0)

                        processed_frames += 1
                        logger.info(
                            f"Board changed: {last_count}→{current_count} stones "
                            f"(+{stones_added}, -{stones_removed}) "
                            f"at frame {frame_count}"
                        )

                        _, _ = go_game.main_loop(frame, end_game=False)
                        last_board_state = current_board.copy()
                        last_analysis_time = current_time

                        if processed_frames % 5 == 0:
                            logger.info(f"Processed {processed_frames} states... "
                                      f"(frame {frame_count}/{total_frames})")

                    elif stones_removed > stones_added:
                        # More stones removed than added - likely correction
                        current_count = np.sum(current_board > 0)
                        last_count = np.sum(last_board_state > 0)
                        logger.warning(
                            f"Stone removal detected: {last_count}→{current_count} "
                            f"(+{stones_added}, -{stones_removed}) at frame {frame_count}. "
                            "Updating baseline without saving."
                        )
                        # Update baseline but don't save to history
                        last_board_state = current_board.copy()
                        last_analysis_time = current_time

                    else:
                        logger.debug(f"No meaningful change at frame {frame_count}")

            except Exception as e:
                logger.debug(f"Verification failed at frame {frame_count}: {e}")

    logger.info(f"Processing complete. Analyzed {processed_frames} states.")
    return processed_frames


def process_video_fixed_interval(cap: cv2.VideoCapture,
                                 go_game: GoGame,
                                 interval: float) -> int:
    """Process video at fixed intervals (original method)."""
    logger.info("Processing video with fixed interval...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.warning("Video FPS is 0. Defaulting to 30.")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval))

    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}, "
                f"Analyzing every {frame_interval} frames "
                f"(interval: {interval}s)")

    processed_frames = 0
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

    mode = "TRANSPARENT (AI Post-Processing)" if use_transparent else "REAL-TIME"
    logger.info(f"Running in {mode} mode")

    logger.info(f"Opening video file: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Could not open video file {args.video}")
        return

    # Initialize Board
    if not initialize_board(cap, go_game):
        cap.release()
        return

    # Process Video
    if args.fixed_interval:
        processed_frames = process_video_fixed_interval(
            cap, go_game, args.analysis_interval
        )
    else:
        processed_frames = process_video_motion_detection(
            cap, go_game, args.min_analysis_interval
        )

    cap.release()

    # Post-Process and Save SGF
    final_sgf = None
    if use_transparent:
        num_states = len(go_game.numpy_board)
        logger.info(f"Running AI post-processing on {num_states} board states...")

        if num_states < 2:
            logger.error("Not enough board states captured for AI processing.")
        else:
            try:
                final_sgf = go_game.post_treatment(end_game=True)
                if final_sgf:
                    logger.info(f"Generated SGF with {len(final_sgf)} characters")
                else:
                    logger.warning("Empty SGF generated by AI.")
            except Exception as e:
                logger.error(f"Error during AI post-processing: {e}", exc_info=True)
                logger.info("Attempting fallback SGF generation (no AI)...")
                try:
                    move_list = corrector_no_ai(go_game.numpy_board)
                    final_sgf = to_sgf(move_list)
                    logger.info("Fallback SGF generation successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    final_sgf = None
    else:
        final_sgf = go_game.get_sgf()

    # Save File
    if final_sgf:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            with open(args.output, "w") as f:
                f.write(final_sgf)
            logger.info(f"\n✓ Successfully saved game to {args.output}")
            logger.info(f"  Total frames analyzed: {processed_frames}")
        except IOError as e:
            logger.error(f"\n✗ Error: Could not write SGF to {args.output}: {e}")
    else:
        logger.error("\n✗ Error: No SGF data was generated.")


if __name__ == "__main__":
    arguments = parse_arguments()
    run_pipeline(arguments)
