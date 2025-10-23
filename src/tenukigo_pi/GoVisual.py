"""
Visual representation and navigation of Go games.
"""

import numpy as np
import cv2
import sente


class GoVisual:
    """
    Manages visual representation of sente games.

    Draws Go board and stones, handles navigation through move history.
    """

    def __init__(self, game):
        """
        Initialize GoVisual.

        Args:
            game: Sente game instance to visualize
        """
        self.game = game
        self.board_size = 19
        self.last_move = None
        self.cursor = len(self.get_moves())
        self.track_progress = True

    def get_stones(self, board):
        """
        Count and collect stone positions from sente numpy board.

        Args:
            board: 19x19x2 array from game.numpy()

        Returns:
            tuple: (black_stones, white_stones) as lists of (row, col)
        """
        black_stones = []
        white_stones = []
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if np.array_equal(board[i, j], [1, 0]):
                    black_stones.append((i, j))
                elif np.array_equal(board[i, j], [0, 1]):
                    white_stones.append((i, j))
        return black_stones, white_stones

    def update_param(self):
        """Update internal sente game state to match cursor position."""
        deleted_moves = []
        current_move_count = len(self.get_moves())
        if self.cursor - current_move_count != 0:
            deleted_moves = self.get_moves()[
                self.cursor - current_move_count:
            ]

        self.game.step_up(current_move_count - self.cursor)
        black_stones, white_stones = self.get_stones(
            self.game.numpy(["black_stones", "white_stones"])
        )

        moves = self.get_moves()
        if moves:
            self.last_move = moves[-1]
        else:
            self.last_move = None

        for move in deleted_moves:
            self.game.play(move.get_x() + 1, move.get_y() + 1)

        return black_stones, white_stones

    def get_moves(self):
        """Get SGF move sequence, filtering out pass moves."""
        moves = []
        for move in self.game.get_sequence():
            if move.get_x() == 19 and move.get_y() == 19:
                continue
            moves.append(move)
        return moves

    def initial_position(self):
        """Set cursor to first move."""
        self.track_progress = False
        self.cursor = 1

    def final_position(self):
        """Set cursor to last move and enable tracking."""
        self.track_progress = True
        self.cursor = len(self.get_moves())

    def current_turn(self):
        """Determine whose turn it is to play."""
        if self.last_move and self.last_move.get_stone().name == 'BLACK':
            return 'WHITE'
        return 'BLACK'

    def previous(self):
        """Move cursor back one move."""
        self.track_progress = False
        if self.cursor > 1:
            self.cursor -= 1

    def next(self):
        """Move cursor forward one move."""
        self.track_progress = False
        moves_count = len(self.get_moves())
        if self.cursor < moves_count:
            self.cursor += 1

        if self.cursor == moves_count:
            self.track_progress = True

    def current_position(self):
        """Draw board at current cursor position."""
        if self.track_progress:
            self.cursor = len(self.get_moves())

        black_stones, white_stones = self.update_param()
        return self.draw_board(black_stones, white_stones)

    def draw_board(self, black_stones, white_stones):
        """
        Render Go board with stones as BGR image.

        Args:
            black_stones: List of (row, col) for black
            white_stones: List of (row, col) for white

        Returns:
            np.array: BGR image of Go board
        """
        square_size = 30
        circle_radius = 12
        board_dim = (self.board_size + 1) * square_size

        board_color = (69, 166, 245)
        board = np.full((board_dim, board_dim, 3), board_color, dtype=np.uint8)

        for i in range(1, self.board_size + 1):
            x = square_size * i
            cv2.line(board, (x, square_size),
                     (x, square_size * self.board_size), (0, 0, 0), 1)
            letter = chr(ord('A') + i - 1)
            cv2.putText(board, letter, (x, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(board, letter, (x, 585),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            y = square_size * i
            cv2.line(board, (square_size, y),
                     (square_size * self.board_size, y), (0, 0, 0), 1)
            cv2.putText(board, str(i), (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(board, str(i), (580, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for stone in black_stones:
            row, col = stone
            center = ((row + 1) * square_size, (col + 1) * square_size)
            cv2.circle(board, center, circle_radius, (66, 66, 66), 2)
            cv2.circle(board, center, circle_radius, (0, 0, 0), -1)

        for stone in white_stones:
            row, col = stone
            center = ((row + 1) * square_size, (col + 1) * square_size)
            cv2.circle(board, center, circle_radius, (66, 66, 66), 2)
            cv2.circle(board, center, circle_radius, (255, 255, 255), -1)

        if self.last_move is not None:
            row, col = self.last_move.get_x(), self.last_move.get_y()
            color = self.last_move.get_stone().name
            stone_color = (0, 0, 0) if color == 'BLACK' else (255, 255, 255)
            center = ((row + 1) * square_size, (col + 1) * square_size)
            cv2.circle(board, center, circle_radius, (0, 0, 255), 2)
            cv2.circle(board, center, circle_radius, stone_color, -1)

        return board

    def load_game_from_sgf(self, sgf_url):
        """Load game from SGF file."""
        self.game = sente.sgf.load(sgf_url)
        self.game.play_sequence(self.game.get_default_sequence())
        return self.current_position()

    def draw_transparent(self, detected_state):
        """Draw board exactly as detected, ignoring game rules."""
        black_stones, white_stones = self.get_stones(detected_state)
        self.last_move = None
        return self.draw_board(black_stones, white_stones)
