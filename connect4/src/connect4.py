import numpy as np
from typing import Optional, Tuple

# Constants
ROWS = 6
COLS = 7
EMPTY = 0
PLAYER1 = 1
PLAYER2 = 2


def new_game() -> np.ndarray:
    """Create a new game state with an empty board."""
    return np.zeros((ROWS, COLS), dtype=np.int8)


def is_valid_move(state: np.ndarray, col: int) -> bool:
    """Check if a move in the given column is valid."""
    if col < 0 or col >= COLS:
        return False
    return state[0, col] == EMPTY


def make_move(state: np.ndarray, col: int, player: int) -> np.ndarray:
    """Make a move and return the new state. Does not modify the original state."""
    if not is_valid_move(state, col):
        raise ValueError(f"Invalid move: column {col}")

    new_state = state.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_state[row, col] == EMPTY:
            new_state[row, col] = player
            break
    return new_state


def get_valid_moves(state: np.ndarray) -> list[int]:
    """Return a list of valid column indices."""
    return [col for col in range(COLS) if is_valid_move(state, col)]


def check_winner(state: np.ndarray) -> Optional[int]:
    """Check if there's a winner. Returns player number (1 or 2) or None."""
    # Check horizontal
    for row in range(ROWS):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if np.all(state[row, col:col+4] == state[row, col]):
                    return state[row, col]

    # Check vertical
    for row in range(ROWS - 3):
        for col in range(COLS):
            if state[row, col] != EMPTY:
                if np.all(state[row:row+4, col] == state[row, col]):
                    return state[row, col]

    # Check diagonal (bottom-left to top-right)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if all(state[row-i, col+i] == state[row, col] for i in range(4)):
                    return state[row, col]

    # Check diagonal (top-left to bottom-right)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if state[row, col] != EMPTY:
                if all(state[row+i, col+i] == state[row, col] for i in range(4)):
                    return state[row, col]

    return None


def is_draw(state: np.ndarray) -> bool:
    """Check if the game is a draw (board full, no winner)."""
    return len(get_valid_moves(state)) == 0 and check_winner(state) is None


def is_terminal(state: np.ndarray) -> bool:
    """Check if the game is over (winner or draw)."""
    return check_winner(state) is not None or is_draw(state)


def state_to_string(state: np.ndarray) -> str:
    """Convert game state to a string representation."""
    chars = {EMPTY: '.', PLAYER1: 'X', PLAYER2: 'O'}
    lines = []
    for row in state:
        lines.append(''.join(chars[cell] for cell in row))
    return '\n'.join(lines)


def string_to_state(s: str) -> np.ndarray:
    """Convert a string representation back to game state."""
    chars = {'.': EMPTY, 'X': PLAYER1, 'O': PLAYER2}
    lines = s.strip().split('\n')
    state = np.zeros((ROWS, COLS), dtype=np.int8)
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            state[i, j] = chars[char]
    return state
