# Imports
import numpy as np
from keras.saving import load_model


def load_corrector_model(model_path):
    """
    Loads the Keras model from the given path.

    Args:
        model_path (str): The file path to the .keras or .h5 model.

    Returns:
        keras.Model: The loaded model.
    """
    print(f"Loading corrector model from: {model_path}")
    # compile=False is crucial for loading models saved with an optimizer
    # when you only need to do inference.
    model = load_model(model_path, compile=False)
    return model


# UTILS
def sgf_coords_to_indices(coord, board_size):
    """Convert SGF coordinates (e.g., 'pd') to array indices (row, col)."""
    col = ord(coord[0]) - ord('a')
    row = ord(coord[1]) - ord('a')
    return board_size - 1 - row, col


def sgf_to_sequence(sgf_file, board_size=19):
    """
    Convert an SGF file to a sequence of Go board states.

    Args:
        sgf_file (str): Path to the SGF file.
        board_size (int): Size of the Go board.

    Returns:
        list: A sequence (list) of 19x19 np.array board states.
    """
    with open(sgf_file, 'r') as f:
        sgf_content = f.read()

    # We need to import sgf here to avoid circular dependencies
    # or make this a separate utility module.
    # For now, local import:
    try:
        from sente import sgf
    except ImportError:
        print("Sente library not found. SGF functions will fail.")
        return []

    collection = sgf.parse(sgf_content)
    game = collection[0]  # Assume a single game
    board = np.zeros((board_size, board_size), dtype=int)
    sequence = [board.copy()]

    for node in game.rest:
        move = node.properties
        if 'B' in move:  # Black move
            x, y = sgf_coords_to_indices(move['B'][0], board_size)
            board[x, y] = 1
        elif 'W' in move:  # White move
            x, y = sgf_coords_to_indices(move['W'][0], board_size)
            board[x, y] = 2
        sequence.append(board.copy())

    return sequence


def sequence_to_sgf(sequence, board_size=19):
    """
    Convert a sequence of Go board states back to SGF format.

    Args:
        sequence (list): Sequence of np.array board states.
        board_size (int): Size of the Go board.

    Returns:
        str: SGF representation of the game.
    """
    sgf_moves = []
    prev_board = np.zeros_like(sequence[0])

    for board in sequence[1:]:
        diff = board - prev_board
        move = np.where(diff > 0)
        if len(move[0]) > 0:  # There is a move
            x, y = move[0][0], move[1][0]
            color = 'B' if board[x, y] == 1 else 'W'
            coords = indices_to_sgf_coords(x, y, board_size)
            sgf_moves.append(f";{color}[{coords}]")
        prev_board = board

    sgf_string = f"(;GM[1]SZ[{board_size}]" + "".join(sgf_moves) + ")"
    return sgf_string


def indices_to_sgf_coords(x, y, board_size):
    """Convert array indices (row, col) to SGF coordinates (e.g., 'pd')."""
    col_char = chr(y + ord('a'))
    row_char = chr(board_size - 1 - x + ord('a'))
    return f"{col_char}{row_char}"


def save_sgf_to_file(sgf_string, file_path):
    """
    Save an SGF string to a file.

    Args:
        sgf_string (str): SGF content to save.
        file_path (str): Path to save the SGF file.
    """
    with open(file_path, 'w') as f:
        f.write(sgf_string)
    print(f"SGF saved to {file_path}")


def delete_states(sequence, start, end):
    """
    Replace states with zeros to create gaps.

    Args:
        sequence (list): Original sequence of Go board states.
        start (int): Starting index of the gap.
        end (int): Ending index of the gap (exclusive).

    Returns:
        list: Sequence with states replaced by zeros.
    """
    board_shape = sequence[0].shape
    for i in range(start, end):
        sequence[i] = np.zeros(board_shape, dtype=int)
    return sequence


def get_possible_moves(initial_state, final_state):
    """
    Get possible moves in a gap by diffing the start and end states.

    Args:
        initial_state (np.array): The board state *before* the gap.
        final_state (np.array): The board state *after* the gap.
    Returns:
        tuple: A tuple containing:
            - list: List of (row, col) tuples for Black moves.
            - list: List of (row, col) tuples for White moves.
    """
    difference = final_state - initial_state

    # Find all black moves (difference == 1)
    black_moves = np.argwhere(difference == 1)
    black_moves = [tuple(move) for move in black_moves]

    # Find all white moves (difference == 2)
    white_moves = np.argwhere(difference == 2)
    white_moves = [tuple(move) for move in white_moves]

    return black_moves, white_moves


# FILL GAPS FUNCTION

def fill_gaps(model, sequence_with_gap, gap_start, gap_end,
              black_possible_moves, white_possible_moves):
    """
    Fill the gaps in a sequence using the AI model to pick the best move.

    Args:
        model: The trained Keras model.
        sequence_with_gap (list): The sequence of board states with gaps.
        gap_start (int): The start index of the gap.
        gap_end (int): The end index of the gap (exclusive).
        black_possible_moves (list): List of (r, c) tuples for Black.
        white_possible_moves (list): List of (r, c) tuples for White.

    Returns:
        list: The sequence with the gaps filled.
    """
    filled_sequence = sequence_with_gap.copy()

    # Determine current player based on the last move *before* the gap.
    state_before_gap_1 = sequence_with_gap[gap_start - 1]
    state_before_gap_2 = sequence_with_gap[gap_start - 2]
    difference = state_before_gap_1 - state_before_gap_2

    # If diff=1, Black just played, so current_player is White (2).
    # Otherwise, it's Black's turn (1).
    current_player = 2 if np.any(difference == 1) else 1

    black_moves = black_possible_moves.copy()
    white_moves = white_possible_moves.copy()

    for gap_index in range(gap_start, gap_end):
        current_board_state = filled_sequence[gap_index - 1]
        possible_moves = black_moves if current_player == 1 else white_moves

        # Find moves that are valid (i.e., on an empty intersection)
        valid_moves = [
            move for move in possible_moves
            if current_board_state[move[0], move[1]] == 0
        ]

        candidate_boards = []
        candidate_moves = []

        for move in valid_moves:
            x, y = move
            candidate_board = current_board_state.copy()
            candidate_board[x, y] = current_player
            candidate_boards.append(candidate_board)
            candidate_moves.append(move)

        if not candidate_boards:
            print(f"No valid moves for gap index {gap_index}, skipping.")
            # We must copy the previous state to not have an empty board
            filled_sequence[gap_index] = current_board_state.copy()
            continue

        # Prepare batch for the model
        candidate_boards = np.array(candidate_boards)
        candidate_boards = np.expand_dims(candidate_boards, axis=-1)
        candidate_boards = candidate_boards.astype(np.float32)

        # Predict probabilities for all candidate boards at once
        probabilities = model.predict(candidate_boards)

        # Get the index of the best move
        # probabilities[:, 0] is prob of Black, [:, 1] is prob of White
        best_move_idx = np.argmax(probabilities[:, current_player - 1])
        best_move = candidate_moves[best_move_idx]

        # Update the board state in the sequence
        x, y = best_move
        filled_sequence[gap_index] = current_board_state.copy()
        filled_sequence[gap_index][x, y] = current_player

        # Remove the chosen move from the list of possibilities
        if current_player == 1:
            black_moves.remove(best_move)
        else:
            white_moves.remove(best_move)

        print(f"Filling gap {gap_index}: Player {current_player} at {best_move}")

        # Switch player for the next move
        current_player = 3 - current_player  # 1 -> 2, 2 -> 1

    return filled_sequence
