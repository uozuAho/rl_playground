import numpy as np
import typing as t

ROWS = 6
COLS = 7
EMPTY = 0
type Player = t.Literal[1, -1]
PLAYER1: Player = 1
PLAYER2: Player = -1


class GameState:
    board: np.ndarray
    current_player: Player
    done: bool
    winner: Player | None

    @staticmethod
    def new():
        s = GameState()
        s.board = np.zeros((ROWS, COLS), dtype=np.int8)
        s.current_player = PLAYER1
        s.done = False
        s.winner = None
        return s

    def copy(self):
        s = GameState()
        s.board = self.board.copy()
        s.current_player = self.current_player
        s.done = self.done
        s.winner = self.winner
        return s

    def equals(self, other: GameState):
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
    for row in range(ROWS - 1, -1, -1):
        if new_state.board[row, col] == EMPTY:
            new_state.board[row, col] = state.current_player
            break

    new_state.winner = calc_winner(new_state)
    new_state.done = (
        new_state.winner is not None or len(get_valid_moves(new_state)) == 0
    )
    new_state.current_player = other_player(state.current_player)

    return new_state


def get_valid_moves(state: GameState) -> list[int]:
    return [col for col in range(COLS) if is_valid_move(state, col)]


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


# todo: replace this by reading done
def is_terminal(state: GameState) -> bool:
    return calc_winner(state) is not None or is_draw(state)


def to_string(state: GameState) -> str:
    chars = {EMPTY: ".", PLAYER1: "X", PLAYER2: "O"}
    lines = []
    for row in state.board:
        lines.append("".join(chars[cell] for cell in row))
    return "\n".join(lines)


def from_string(s: str) -> GameState:
    chars = {".": EMPTY, "X": PLAYER1, "O": PLAYER2}
    lines = s.strip().split("\n")
    state = new_game()
    numx = s.count("X")
    numo = s.count("O")
    assert numx - numo in [0, 1]
    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            state.board[i, j] = chars[char]
    state.current_player = PLAYER1 if numx == numo else PLAYER2
    state.done = len(get_valid_moves(state)) == 0 and calc_winner(state) is None
    return state


def other_player(player: Player):
    return PLAYER2 if player == PLAYER1 else PLAYER1


def are_equal(s1: GameState, s2: GameState):
    return s1.equals(s2)
