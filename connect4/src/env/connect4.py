import numpy as np
import typing as t

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = -1

type GameState = np.ndarray
type Player = t.Literal[1, -1]


def new_game() -> GameState:
    return np.zeros((ROWS, COLS), dtype=np.int8)


def is_valid_move(state: GameState, col: int) -> bool:
    if col < 0 or col >= COLS:
        return False
    return state[0, col] == EMPTY


def make_move(state: GameState, col: int, player: int) -> GameState:
    if not is_valid_move(state, col):
        raise ValueError(f"Invalid move: column {col}")

    new_state = state.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_state[row, col] == EMPTY:
            new_state[row, col] = player
            break
    return new_state


def get_valid_moves(state: GameState) -> list[int]:
    """Return a list of valid column indices."""
    return [col for col in range(COLS) if is_valid_move(state, col)]


def winner(state: GameState) -> Player | None:
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if np.all(state[row, col : col + 4] == state[row, col]):
                    return state[row, col]

    # Check vertical
    for row in range(ROWS - 3):
        for col in range(COLS):
            if state[row, col] != EMPTY:
                if np.all(state[row : row + 4, col] == state[row, col]):
                    return state[row, col]

    # Check diagonal (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if all(state[row - i, col + i] == state[row, col] for i in range(4)):
                    return state[row, col]

    # Check diagonal (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if all(state[row + i, col + i] == state[row, col] for i in range(4)):
                    return state[row, col]

    return None


def is_draw(state: GameState) -> bool:
    return len(get_valid_moves(state)) == 0 and winner(state) is None


def is_terminal(state: GameState) -> bool:
    return winner(state) is not None or is_draw(state)


def to_string(state: GameState) -> str:
    chars = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
    lines = []
    for row in state:
        lines.append("".join(chars[cell] for cell in row))
    return "\n".join(lines)


def from_string(s: str) -> np.ndarray:
    chars = {".": EMPTY, "X": PLAYER1, "O": PLAYER2}
    lines = s.strip().split("\n")
    state = np.zeros((ROWS, COLS), dtype=np.int8)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            state[i, j] = chars[char]
    return state


def other_player(player: Player):
    return PLAYER2 if player == PLAYER1 else PLAYER1
