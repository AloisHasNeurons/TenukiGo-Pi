import sente
from sente import sgf
import numpy as np


def sgf_to_numpy(sgf_file_path):
    """
    Converts an SGF file into a sequence of numpy arrays.

    Args:
        sgf_file_path (str): The file path to the .sgf file.

    Returns:
        np.array: A NumPy array of shape (num_moves + 1, 19, 19),
                  where:
                  - 0 = empty
                  - 1 = black stone
                  - 2 = white stone
    """
    game = sgf.load(sgf_file_path)
    moves = game.get_default_sequence()
    num_moves = len(moves)
    # result[i] = board state at move i (index 0 is empty board)
    result = np.zeros((num_moves + 1, 19, 19), dtype=int)

    for i in range(1, num_moves + 1):
        game.play(moves[i - 1])
        # Get sente's 19x19x1 numpy arrays
        black_stones_np = game.numpy(["black_stones"])
        white_stones_np = game.numpy(["white_stones"])

        # Transpose and fill our result array
        # Sente's numpy is (col, row, channel), we want (row, col)
        for row in range(19):
            for col in range(19):
                if black_stones_np[col][row][0] == 1:
                    result[i, row, col] = 1
                elif white_stones_np[col][row][0] == 1:
                    result[i, row, col] = 2
    return result


def to_sgf(move_list):
    """
    Converts a simple list of moves into an SGF file string.

    Args:
        move_list (list): A list of move tuples, where each tuple is
                          (row, col, player_num).
                          - player_num: 1 for Black, 2 for White.
                          - row, col: 0-18 indices.

    Returns:
        str: A string containing the SGF data.
    """
    game = sente.Game()
    for move in move_list:
        row, col, player = move
        # Sente uses 1-19 indexing for play()
        game.play(row + 1, col + 1)
    return sgf.dumps(game)
