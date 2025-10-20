import numpy as np
import cv2
import sente


class GoVisual:
    """
    Manages the visual representation of a `sente` game.

    This class draws the Go board and stones onto a NumPy array (image)
    and handles navigation through the game's move history (next, previous,
    first, last).

    Attributes:
        game (sente.Game): The `sente` game instance to visualize.
        board_size (int): The size of the board (default 19).
        last_move (sente.Move): The last move played.
        cursor (int): The index of the move currently being displayed.
        track_progress (bool): If True, the visualizer automatically
                               jumps to the latest move.
    """

    def __init__(self, game):
        """
        Initializes the GoVisual class.

        Args:
            game (sente.Game): The game instance created by Sente, which
                               is shared with and updated by GoGame.
        """
        self.game = game
        self.board_size = 19
        self.last_move = None
        self.cursor = len(self.get_moves())
        self.track_progress = True

    def get_stones(self, board):
        """
        Counts and collects positions of stones from a `sente` numpy board.

        Args:
            board (np.array): A 19x19x2 array from `game.numpy()`.

        Returns:
            tuple: (black_stones, white_stones) as lists of (row, col) tuples.
        """
        black_stones = []
        white_stones = []
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if np.array_equal(board[i, j], [1, 0]):  # Black
                    black_stones.append((i, j))
                elif np.array_equal(board[i, j], [0, 1]):  # White
                    white_stones.append((i, j))
        return black_stones, white_stones

    def update_param(self):
        """
        Updates the internal state of the `sente` game to match the cursor.

        This rewinds and fast-forwards the `sente.Game` object to show
        the board state at the `self.cursor` move index.
        """
        deleted_moves = []
        current_move_count = len(self.get_moves())
        if self.cursor - current_move_count != 0:
            deleted_moves = self.get_moves()[
                self.cursor - current_move_count:
            ]

        # Rewind the game to the cursor position
        self.game.step_up(current_move_count - self.cursor)
        black_stones, white_stones = self.get_stones(
            self.game.numpy(["black_stones", "white_stones"])
        )

        moves = self.get_moves()
        if moves:
            self.last_move = moves[-1]
        else:
            self.last_move = None

        # Re-play any moves that were "undone" to get back to the true
        # end of the game tree, allowing `sente` to accept new moves.
        for move in deleted_moves:
            self.game.play(move.get_x() + 1, move.get_y() + 1)

        return black_stones, white_stones

    def get_moves(self):
        """
        Gets the SGF move sequence, filtering out "pass" moves.
        `sente` represents pass moves as (19, 19).

        Returns:
            list: The cleaned list of `sente.Move` objects.
        """
        moves = []
        for move in self.game.get_sequence():
            # A move at (19, 19) is a "pass" in sente
            if move.get_x() == 19 and move.get_y() == 19:
                continue
            moves.append(move)
        return moves

    def initial_position(self):
        """Sets the cursor to the first move."""
        self.track_progress = False
        self.cursor = 1

    def final_position(self):
        """Sets the cursor to the last move and enables tracking."""
        self.track_progress = True
        self.cursor = len(self.get_moves())

    def current_turn(self):
        """
        Determines whose turn it is to play.

        Returns:
            str: "BLACK" or "WHITE".
        """
        if self.last_move and self.last_move.get_stone().name == 'BLACK':
            return 'WHITE'
        else:
            # White played last, or it's the start of the game
            return 'BLACK'

    def previous(self):
        """Moves the cursor back one move."""
        self.track_progress = False
        if self.cursor > 1:
            self.cursor -= 1

    def next(self):
        """Moves the cursor forward one move."""
        self.track_progress = False
        moves_count = len(self.get_moves())
        if self.cursor < moves_count:
            self.cursor += 1

        if self.cursor == moves_count:
            self.track_progress = True

    def current_position(self):
        """
        Draws the board at the current cursor position.

        If tracking is enabled, automatically moves cursor to the latest move.

        Returns:
            np.array: A BGR image of the Go board.
        """
        if self.track_progress:
            self.cursor = len(self.get_moves())

        black_stones, white_stones = self.update_param()
        return self.draw_board(black_stones, white_stones)

    def draw_board(self, black_stones, white_stones):
        """
        Renders the Go board with stones as a BGR image.

        Args:
            black_stones (list): List of (row, col) tuples for black.
            white_stones (list): List of (row, col) tuples for white.

        Returns:
            np.array: A BGR image of the Go board.
        """
        square_size = 30
        circle_radius = 12
        board_dim = (self.board_size + 1) * square_size

        # Set up the board's background
        board_color = (69, 166, 245)  # Brownish color
        board = np.full((board_dim, board_dim, 3), board_color, dtype=np.uint8)

        for i in range(1, self.board_size + 1):
            # Vertical lines and letters (A-T)
            x = square_size * i
            cv2.line(board, (x, square_size),
                     (x, square_size * self.board_size), (0, 0, 0), 1)
            letter = chr(ord('A') + i - 1)
            cv2.putText(board, letter, (x, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(board, letter, (x, 585),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Horizontal lines and numbers (1-19)
            y = square_size * i
            cv2.line(board, (square_size, y),
                     (square_size * self.board_size, y), (0, 0, 0), 1)
            cv2.putText(board, str(i), (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(board, str(i), (580, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw stones
        for stone in black_stones:
            row, col = stone
            center = ((row + 1) * square_size, (col + 1) * square_size)
            cv2.circle(board, center, circle_radius, (66, 66, 66), 2)  # Edge
            cv2.circle(board, center, circle_radius, (0, 0, 0), -1)  # Fill

        for stone in white_stones:
            row, col = stone
            center = ((row + 1) * square_size, (col + 1) * square_size)
            cv2.circle(board, center, circle_radius, (66, 66, 66), 2)  # Edge
            cv2.circle(board, center, circle_radius, (255, 255, 255), -1)

        # Highlight the last move
        if self.last_move is not None:
            row, col = self.last_move.get_x(), self.last_move.get_y()
            color = self.last_move.get_stone().name
            stone_color = (0, 0, 0) if color == 'BLACK' else (255, 255, 255)
            center = ((row + 1) * square_size, (col + 1) * square_size)
            # Red circle for highlight
            cv2.circle(board, center, circle_radius, (0, 0, 255), 2)
            cv2.circle(board, center, circle_radius, stone_color, -1)

        return board

    def load_game_from_sgf(self, sgf_url):
        """
        Loads a game from an SGF file.

        Args:
            sgf_url (str): The file path of the SGF file.

        Returns:
            np.array: A BGR image of the final board position.
        """
        self.game = sente.sgf.load(sgf_url)
        self.game.play_sequence(self.game.get_default_sequence())
        return self.current_position()

    def draw_transparent(self, detected_state):
        """
        Draws the board exactly as detected, ignoring game rules.
        Used for "transparent mode".

        Args:
            detected_state (np.array): The 19x19x2 board state.

        Returns:
            np.array: A BGR image of the board.
        """
        black_stones, white_stones = self.get_stones(detected_state)
        self.last_move = None  # No "last move" in this mode
        return self.draw_board(black_stones, white_stones)
