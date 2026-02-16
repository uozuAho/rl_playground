import textwrap

import numpy as np
import typing as t

ROWS = 6
COLS = 7
EMPTY = 0
type Player = t.Literal[1, -1]
type Action = int  # put piece in col x, 0-6
ACTION_SIZE = COLS
PLAYER1: Player = 1
PLAYER2: Player = -1


class GameState:
    board: np.ndarray
    current_player: Player
    done: bool
    winner: Player | None
    moves_played: int = 0

    @staticmethod
    def new():
        s = GameState()
        s.board = np.zeros((ROWS, COLS), dtype=np.int8)
        s.current_player = PLAYER1
        s.done = False
        s.winner = None
        return s

    def __repr__(self):
        cp_str = "X" if self.current_player == PLAYER1 else "O"
        winner_str = (
            " " if self.winner is None else "X" if self.winner == PLAYER1 else "O"
        )
        return (
            f"cp: {cp_str}, done: {self.done}, winner: {winner_str}, {to_string(self)}"
        )

    def copy(self):
        s = GameState()
        s.board = self.board.copy()
        s.current_player = self.current_player
        s.done = self.done
        s.winner = self.winner
        s.moves_played = self.moves_played
        return s

    def equals(self, other: "GameState"):
        return (
            np.array_equal(self.board, other.board)
            and self.current_player == other.current_player
            and self.done == other.done
            and self.winner == other.winner
        )


def new_game() -> GameState:
    return GameState.new()


def is_valid_move(state: GameState, col: int) -> bool:
    if col < 0 or col >= COLS:
        return False
    return state.board[0, col] == EMPTY


def make_move(state: GameState, col: int) -> GameState:
    if not is_valid_move(state, col):
        raise ValueError(f"Invalid move: column {col}")

    new_state = state.copy()
    new_state.moves_played += 1
    placed_row = -1
    for row in range(ROWS - 1, -1, -1):
        if new_state.board[row, col] == EMPTY:
            new_state.board[row, col] = state.current_player
            placed_row = row
            break

    if new_state.moves_played > 6:
        new_state.winner = calc_winner_at_position(new_state, placed_row, col)
        new_state.done = (
            new_state.winner is not None or len(get_valid_moves(new_state)) == 0
        )
    new_state.current_player = other_player(state.current_player)

    return new_state


def get_valid_moves(state: GameState) -> list[int]:
    return [col for col in range(COLS) if is_valid_move(state, col)]


def calc_winner_at_position(state: GameState, row: int, col: int) -> Player | None:
    """Check if placing a piece at (row, col) creates a winner by checking 4 directions."""
    player = state.board[row, col]
    if player == EMPTY:
        return None

    # Check horizontal
    count = 1
    # Check left
    for c in range(col - 1, -1, -1):
        if state.board[row, c] == player:
            count += 1
        else:
            break
    # Check right
    for c in range(col + 1, COLS):
        if state.board[row, c] == player:
            count += 1
        else:
            break
    if count >= 4:
        return player

    # Check vertical
    count = 1
    # Check up
    for r in range(row - 1, -1, -1):
        if state.board[r, col] == player:
            count += 1
        else:
            break
    # Check down
    for r in range(row + 1, ROWS):
        if state.board[r, col] == player:
            count += 1
        else:
            break
    if count >= 4:
        return player

    # Check diagonal (top-left to bottom-right)
    count = 1
    # Check up-left
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0:
        if state.board[r, c] == player:
            count += 1
            r -= 1
            c -= 1
        else:
            break
    # Check down-right
    r, c = row + 1, col + 1
    while r < ROWS and c < COLS:
        if state.board[r, c] == player:
            count += 1
            r += 1
            c += 1
        else:
            break
    if count >= 4:
        return player

    # Check diagonal (bottom-left to top-right)
    count = 1
    # Check down-left
    r, c = row + 1, col - 1
    while r < ROWS and c >= 0:
        if state.board[r, c] == player:
            count += 1
            r += 1
            c -= 1
        else:
            break
    # Check up-right
    r, c = row - 1, col + 1
    while r >= 0 and c < COLS:
        if state.board[r, c] == player:
            count += 1
            r -= 1
            c += 1
        else:
            break
    if count >= 4:
        return player

    return None


def calc_winner(state: GameState) -> Player | None:
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            if state.board[row, col] != EMPTY:
                if np.all(state.board[row, col : col + 4] == state.board[row, col]):
                    return state.board[row, col]

    # Check vertical
    for row in range(ROWS - 3):
        for col in range(COLS):
            if state.board[row, col] != EMPTY:
                if np.all(state.board[row : row + 4, col] == state.board[row, col]):
                    return state.board[row, col]

    # Check diagonal (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if state.board[row, col] != EMPTY:
                if all(
                    state.board[row - i, col + i] == state.board[row, col]
                    for i in range(4)
                ):
                    return state.board[row, col]

    # Check diagonal (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if state.board[row, col] != EMPTY:
                if all(
                    state.board[row + i, col + i] == state.board[row, col]
                    for i in range(4)
                ):
                    return state.board[row, col]

    return None


def is_draw(state: GameState) -> bool:
    return len(get_valid_moves(state)) == 0 and calc_winner(state) is None


def to_string(state: GameState, sep="\n") -> str:
    chars = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
    lines = []
    for row in state.board:
        lines.append("".join(chars[cell] for cell in row))
    return sep.join(lines)


def from_string(s: str) -> GameState:
    chars = {".": EMPTY, "X": PLAYER1, "O": PLAYER2}
    s = textwrap.dedent(s)
    lines = s.strip().split("\n")
    state = new_game()
    numx = s.count("X")
    numo = s.count("O")
    assert numx - numo in [0, 1]
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            state.board[i, j] = chars[char]
    state.current_player = PLAYER1 if numx == numo else PLAYER2
    state.moves_played = numx + numo
    state.winner = calc_winner(state)
    state.done = state.winner is not None or len(get_valid_moves(state)) == 0
    return state


def other_player(player: Player):
    return PLAYER2 if player == PLAYER1 else PLAYER1


def are_equal(s1: GameState, s2: GameState):
    return s1.equals(s2)
