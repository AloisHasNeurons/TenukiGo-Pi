import sente
import numpy as np
# import copy
# from .GoVisual import GoVisual
# from .GoBoard import GoBoard
# from .corrector_noAI import corrector_no_ai
from .corrector_withAI import corrector_with_ai
from .sgf_to_numpy import to_sgf


class GoGame:
    """
    Manages the game logic, state, and move detection.

    This class orchestrates the board detection (`GoBoard`), the game
    state (`sente`), and the visual representation (`GoVisual`). It compares
    frames to find new moves and can use AI to correct errors.

    Attributes:
        board_detect (GoBoard): The board detection object.
        go_visual (GoVisual): The visual representation object.
        game (sente.Game): The core `sente` game logic object.
        corrector_model: The loaded Keras model for AI correction.
        transparent_mode (bool): If True, just records board states without
                                 applying game logic (for post-processing).
        numpy_board (list): A list of 19x19 board states, stored only
                            in transparent mode for post-processing.
    """

    def __init__(self, game, board_detect, go_visual, corrector_model,
                 transparent_mode=False):
        """
        Initializes the GoGame manager.

        Args:
            game (sente.Game): The `sente` game instance.
            board_detect (GoBoard): The initialized `GoBoard` object.
            go_visual (GoVisual): The initialized `GoVisual` object.
            corrector_model: The pre-loaded Keras AI model.
            transparent_mode (bool): Enables transparent mode.
        """
        self.moves = []
        self.board_detect = board_detect
        self.go_visual = go_visual
        self.game = game
        self.corrector_model = corrector_model
        self.current_player = None
        self.transparent_mode = transparent_mode
        self.recent_moves_buffer = []
        self.buffer_size = 5
        self.numpy_board = []
        self.frame = None

    def set_transparent_mode(self, mode):
        """Sets the transparent mode on or off."""
        self.transparent_mode = mode

    def initialize_game(self, frame, current_player="BLACK", end_game=False):
        """
        Initializes the game state from a single frame.

        Resets moves, processes the frame, and auto-populates the board
        based on detected stones.

        Args:
            frame (np.array): The video frame to initialize from.
            current_player (str): "BLACK" or "WHITE".
            end_game (bool): Flag for post-processing logic.

        Returns:
            tuple: (image_to_display, sgf_text)
        """
        self.moves = []
        self.current_player = current_player
        self.frame = frame

        self.board_detect.process_frame(frame)

        if self.transparent_mode:
            detected_state = self.transparent_mode_moves()
            return self.go_visual.draw_transparent(
                detected_state
            ), self.post_treatment(end_game)
        else:
            self.auto_play_game_moves()
            if not self.game.get_active_player().name == current_player:
                self.game.pss()
            return self.go_visual.current_position(), self.get_sgf()

    def main_loop(self, frame, end_game=False):
        """
        Processes a single frame and updates the game state.

        Args:
            frame (np.array): The video frame to process.
            end_game (bool): Flag for post-processing logic.

        Returns:
            tuple: (image_to_display, sgf_text)
        """
        self.frame = frame
        self.board_detect.process_frame(frame)

        if self.transparent_mode:
            detected_state = self.transparent_mode_moves()
            return self.go_visual.draw_transparent(
                detected_state
            ), self.post_treatment(end_game)
        else:
            self.define_new_move()
            return self.go_visual.current_position(), self.get_sgf()

    def copy_board_to_numpy(self):
        """
        Converts the 19x19x2 state to a 19x19 array (0, 1, 2) and
        stores it if it's different from the last stored state.
        Used only in transparent mode.
        """
        final_board = np.zeros((19, 19), dtype=int)
        state = self.board_detect.get_state()
        final_board[state[:, :, 0] == 1] = 1  # 1 for black
        final_board[state[:, :, 1] == 1] = 2  # 2 for white

        if not self.numpy_board or np.any(final_board != self.numpy_board[-1]):
            self.numpy_board.append(final_board)

    def transparent_mode_moves(self):
        """
        Retrieves the current board state and records it for post-processing.

        Returns:
            np.array: The current 19x19x2 board state.
        """
        self.copy_board_to_numpy()
        return np.transpose(self.board_detect.get_state(), (1, 0, 2))

    def play_move(self, x, y, stone_color):
        """
        Plays a move in the `sente` game engine.

        Args:
            x (int): The x-coordinate (1-19).
            y (int): The y-coordinate (1-19).
            stone_color (int): 1 for black, 2 for white.
        """
        color = "white" if stone_color == 2 else "black"
        try:
            self.game.play(x, y, sente.stone(stone_color))
        except sente.exceptions.IllegalMoveException as e:
            err = f"[GoGame] Illegal move at ({x}, {y}): {e}"
            if "self-capture" in str(e):
                raise Exception(err + f" --> {color} self-capture")
            if "occupied point" in str(e):
                raise Exception(err + " --> occupied point")
            if "Ko point" in str(e):
                raise Exception(err + " --> Ko violation")
            if "turn" in str(e):
                raise Exception(err + f" --> Not {color}'s turn")
            raise Exception(err)

    def define_new_move(self):
        """
        Finds the difference between the last known state and the current
        frame's state, then plays the new move.
        """
        detected_state = np.transpose(self.board_detect.get_state(), (1, 0, 2))
        current_state = self.game.numpy(["black_stones", "white_stones"])
        difference = detected_state - current_state

        # Find coordinates of added/removed stones
        black_added = np.argwhere(difference[:, :, 0] == 1)
        white_added = np.argwhere(difference[:, :, 1] == 1)
        black_removed = np.argwhere(difference[:, :, 0] == -1)
        white_removed = np.argwhere(difference[:, :, 1] == -1)

        # Handle multiple moves at once (e.g., fast playing)
        if len(black_added) + len(white_added) > 1:
            self.process_multiple_moves(black_added, white_added)
            return

        # Handle a single new black stone
        if len(black_added) != 0:
            if len(black_added) != 0 and len(black_removed) != 0:
                # Stone was moved, not added. Step back to modify history.
                self.game.step_up()
                print("A stone has been moved")
            x, y = black_added[0][0] + 1, black_added[0][1] + 1
            self.play_move(x, y, 1)  # 1 = black
            self.moves.append(('B', (x - 1, 18 - (y - 1))))
            self.recent_moves_buffer.append({'color': 'B',
                                             'position': black_added[0]})
            self.trim_buffer()

        # Handle a single new white stone
        if len(white_added) != 0:
            if len(white_added) == 1 and len(white_removed) == 1:
                # Stone was moved
                self.game.step_up()
                print("A stone has been moved")
            x, y = white_added[0][0] + 1, white_added[0][1] + 1
            self.play_move(x, y, 2)  # 2 = white
            self.moves.append(('W', (x - 1, 18 - (y - 1))))
            self.recent_moves_buffer.append({'color': 'W',
                                             'position': white_added[0]})
            self.trim_buffer()

    def trim_buffer(self):
        """Ensures the recent moves buffer does not exceed its max size."""
        if len(self.recent_moves_buffer) > self.buffer_size:
            self.recent_moves_buffer.pop(0)

    def process_multiple_moves(self, black_stones, white_stones):
        """Handles the case where multiple stones are added in one frame."""
        for stone in black_stones:
            x, y = stone[0] + 1, stone[1] + 1
            self.play_move(x, y, 1)
            self.moves.append(('B', (x - 1, 18 - (y - 1))))

        for stone in white_stones:
            x, y = stone[0] + 1, stone[1] + 1
            self.play_move(x, y, 2)
            self.moves.append(('W', (x - 1, 18 - (y - 1))))

    def auto_play_game_moves(self):
        """
        Populates the board with all detected stones on initialization.
        This assumes an in-progress game is being loaded.
        """
        detected_state = np.transpose(self.board_detect.get_state(), (1, 0, 2))
        black_stones = np.argwhere(detected_state[:, :, 0] == 1)
        white_stones = np.argwhere(detected_state[:, :, 1] == 1)

        # Play all black stones, with a pass in between
        for stone in black_stones:
            self.play_move(stone[0] + 1, stone[1] + 1, 1)
            self.game.pss()
        self.game.pss()

        # Play all white stones, with a pass in between
        for stone in white_stones:
            self.play_move(stone[0] + 1, stone[1] + 1, 2)
            self.game.pss()
        self.game.pss()

    def correct_stone(self, old_pos, new_pos):
        """
        Manually correct the position of a misplaced stone in the game tree.

        Args:
            old_pos (str): The SGF coordinate of the stone to move (e.g., "A1")
            new_pos (str): The SGF coordinate to move it to (e.g., "S19")
        """
        # Convert SGF coords to 1-19 sente coords
        old_x = int(ord(str(old_pos[0]).upper()) - 64)
        old_y = int(old_pos[1:])
        new_x = int(ord(str(new_pos[0]).upper()) - 64)
        new_y = int(new_pos[1:])

        all_moves = self.get_moves()
        for i in range(len(all_moves)):
            move = all_moves[i]
            # Check if new position is already occupied
            if (move.get_x() + 1) == new_x and (move.get_y() + 1) == new_y:
                print("This position is already occupied!")
                return

            # Find the old move
            if (move.get_x() + 1) == old_x and (move.get_y() + 1) == old_y:
                print("Found stone to correct!")
                # Get all moves *after* the one we're changing
                deleted_moves = all_moves[i - len(all_moves):]
                # Rewind the game back to that point
                self.game.step_up(len(all_moves) - i)
                # Play the *correct* move
                self.game.play(new_x, new_y)
                # Remove the bad move from our temp list
                deleted_moves.pop(0)
                # Re-play all subsequent moves
                for m in deleted_moves:
                    self.game.play(m.get_x() + 1, m.get_y() + 1)
                return

    def delete_last_move(self):
        """Rewinds the game by one move."""
        self.game.step_up()

    def get_moves(self):
        """
        Gets the SGF move sequence, filtering out "pass" moves.
        `sente` represents pass moves as (19, 19).

        Returns:
            list: The cleaned list of `sente.Move` objects.
        """
        moves = []
        for move in self.game.get_sequence():
            if move.get_x() == 19 and move.get_y() == 19:
                continue
            moves.append(move)
        return moves

    def get_sgf(self):
        """
        Gets the SGF (Smart Game Format) string for the current game.

        Returns:
            str: The SGF game data.
        """
        return sente.sgf.dumps(self.game)

    def post_treatment(self, end_game):
        """
        Post-processes the game to correct the move sequence using AI.
        This is only called in transparent mode.

        Args:
            end_game (bool): If True, triggers the correction.

        Returns:
            str: The SGF representation of the *corrected* game.
        """
        if end_game:
            # Use the AI model passed in during __init__
            move_list = corrector_with_ai(self.numpy_board,
                                          self.corrector_model)
            return to_sgf(move_list)
