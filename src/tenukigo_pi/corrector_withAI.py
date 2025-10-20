# import numpy as np
from .sgf_to_numpy import *  # noqa: F403
from .Fill_gaps_model import fill_gaps, get_possible_moves
from .corrector_noAI import differences


def corrector_with_ai(board_states, corrector_model):
    """
    Reconstructs a move list from board states, using an AI model
    to fill gaps when simple heuristics fail.

    Args:
        board_states (list): A list of 19x19 numpy arrays representing
                             the board at each frame.
        corrector_model (keras.Model): The pre-loaded Keras model
                                       used for gap filling.

    Returns:
        list: A list of moves, where each move is a tuple
              (row, col, player_num).
    """
    move_list = []
    # move_list[i] = (row, col, player_num)
    # player_num: 1 for Black, 2 for White
    num_frames = len(board_states)

    turn = 1  # 1 = Black's turn, 2 = White's turn
    not_turn = 2
    index = 1

    while index < num_frames:
        diff_data, num_added = differences(board_states[index - 1],
                                           board_states[index])

        if num_added == 0:
            # No stones added, likely a capture or no change.
            index += 1
            continue

        added_turn_player = diff_data[turn]["add"]
        added_not_turn_player = diff_data[not_turn]["add"]

        # CASE 1: A single, simple move was made by the correct player.
        if len(added_turn_player) == 1 and len(added_not_turn_player) == 0:
            move = added_turn_player[0]
            move_list.append(move)
            print(f"Player {turn} played at {move}")

            # Swap turns for the next iteration
            turn, not_turn = not_turn, turn
            index += 1

        # CASE 2: No moves detected (already handled, but for clarity)
        elif len(added_turn_player) == 0 and len(added_not_turn_player) == 0:
            index += 1
            continue

        # CASE 3: Ambiguous state (e.g., multiple stones, wrong player)
        # This is where the AI model is needed.
        else:
            # We found an ambiguous state. We assume a gap exists between
            # frame [index-1] (last good state) and [index] (bad state).
            # We need to find the *next* good state to define the gap.
            # This implementation assumes the *next* frame [index] is the
            # end of the gap, which is a simplification.
            # A more robust implementation would search forward.

            # Let's re-use the original logic:
            # Insert a copy of the bad frame.
            board_states.insert(index, board_states[index].copy())
            num_frames = len(board_states)

            # Define the gap as being between [index-1] and [index+1]
            b_moves, w_moves = get_possible_moves(board_states[index - 1],
                                                  board_states[index + 1])

            # Call the AI to fill the gap
            board_states = fill_gaps(model=corrector_model,
                                     sequence_with_gap=board_states,
                                     gap_start=index,
                                     gap_end=index + 2,
                                     black_possible_moves=b_moves,
                                     white_possible_moves=w_moves)

            # Update num_frames in case fill_gaps modified the list length
            num_frames = len(board_states)

            if num_frames - 1 == index:
                break

    return move_list
