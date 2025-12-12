"""
AI Model Utilities (Hybrid TFLite/Keras).
Compatible Raspberry Pi (Edge) et PC (Server).
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# --- Environment detection ---
try:
    # Light version for the Raspberry Pi
    import tflite_runtime.interpreter as tflite
    RUNTIME = "TFLITE"
    logger.info("Using TFLite Runtime (Edge optimized)")
except ImportError:
    try:
        import tensorflow.lite as tflite
        RUNTIME = "TFLITE"
        logger.info("Using TensorFlow Lite Interpreter")
    except ImportError:
        # Fallback
        from keras.saving import load_model
        RUNTIME = "KERAS"
        logger.info("Using Full Keras Runtime")


def load_corrector_model(model_path: str):
    """Charge le modèle selon l'environnement disponible."""
    logger.info(f"Loading model from: {model_path}")

    if RUNTIME == "TFLITE":
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return load_model(model_path, compile=False)


def run_inference(model, input_data):
    """Fonction d'inférence agnostique (cache la complexité TFLite)."""
    if RUNTIME == "TFLITE":
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        input_index = input_details[0]['index']
        output_index = output_details[0]['index']

        if input_data.shape != model.get_input_details()[0]['shape']:
            model.resize_tensor_input(input_index, input_data.shape)
            model.allocate_tensors()

        model.set_tensor(input_index, input_data)
        model.invoke()
        return model.get_tensor(output_index)
    else:
        return model.predict(input_data, verbose=0)


def delete_states(sequence: List[np.ndarray], start: int, end: int) -> List[np.ndarray]:
    if not sequence: return []
    board_shape = sequence[0].shape
    for i in range(start, end):
        if i < len(sequence):
            sequence[i] = np.zeros(board_shape, dtype=int)
    return sequence


def get_possible_moves(initial_state: np.ndarray, final_state: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    difference = final_state - initial_state
    black_moves = [tuple(m) for m in np.argwhere(difference == 1)]
    white_moves = [tuple(m) for m in np.argwhere(difference == 2)]
    return black_moves, white_moves


def fill_gaps(model,  # TFlite or Keras
              sequence_with_gap: List[np.ndarray],
              gap_start: int,
              gap_end: int,
              black_possible_moves: List[Tuple[int, int]],
              white_possible_moves: List[Tuple[int, int]]) -> List[np.ndarray]:

    filled_sequence = sequence_with_gap.copy()

    if not (0 <= gap_start < gap_end <= len(filled_sequence)):
        return filled_sequence

    if gap_start >= 2:
        diff = filled_sequence[gap_start - 1] - filled_sequence[gap_start - 2]
        current_player = 2 if np.any(diff == 1) else 1
    else:
        current_player = 1

    black_moves = black_possible_moves.copy()
    white_moves = white_possible_moves.copy()

    for gap_index in range(gap_start, gap_end):
        current_board_state = filled_sequence[gap_index - 1]
        possible_moves = black_moves if current_player == 1 else white_moves

        valid_moves = [m for m in possible_moves if current_board_state[m[0], m[1]] == 0]

        if not valid_moves:
            filled_sequence[gap_index] = current_board_state.copy()
            current_player = 3 - current_player
            continue

        candidate_boards = []
        candidate_moves = []
        for move in valid_moves:
            board = current_board_state.copy()
            board[move[0], move[1]] = current_player
            candidate_boards.append(board)
            candidate_moves.append(move)

        batch_boards = np.array(candidate_boards, dtype=np.float32)
        batch_boards = np.expand_dims(batch_boards, axis=-1)

        try:
            probabilities = run_inference(model, batch_boards)

            best_move_idx = np.argmax(probabilities[:, current_player - 1])
            best_move = candidate_moves[best_move_idx]

            x, y = best_move
            filled_sequence[gap_index] = current_board_state.copy()
            filled_sequence[gap_index][x, y] = current_player

            if current_player == 1:
                if best_move in black_moves: black_moves.remove(best_move)
            else:
                if best_move in white_moves: white_moves.remove(best_move)

        except Exception as e:
            logger.error(f"Prediction error at gap {gap_index}: {e}")
            if valid_moves:
                best_move = valid_moves[0]
                filled_sequence[gap_index] = current_board_state.copy()
                filled_sequence[gap_index][best_move[0], best_move[1]] = current_player

        current_player = 3 - current_player

    return filled_sequence
