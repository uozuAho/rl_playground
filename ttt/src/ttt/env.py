from __future__ import annotations
import typing as t
import gymnasium as gym
from gymnasium import spaces
import numpy as np

type Action = int
type Player = t.Literal[-1, 1]
type Status = t.Literal[-1, 1, 2, 3]
type Board = list[int]

EMPTY = 0
X: Player = 1
O: Player = -1  # noqa: E741
DRAW = 2
IN_PROGRESS = 3

_ACTION_SPACE = spaces.Discrete(9)
_OBS_SPACE = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
_CHAR_LOOKUP = ["NO!", "x", "o"]


class TttEnv:
    def __init__(self):
        self.current_player = X
        self.board = [EMPTY] * 9
        self._step_count = 0
        self.reset()

    def reset(self):
        self.current_player = X
        self.board = [EMPTY] * 9
        self._step_count = 0

    @staticmethod
    def from_str(s: str):
        env = TttEnv()
        for i, c in enumerate(s.replace("|", "").lower()):
            if c == "x":
                env.board[i] = X
            elif c == "o":
                env.board[i] = O
            elif c == ".":
                env.board[i] = EMPTY
            else:
                raise ValueError(f"Invalid character in board string: {c}")
        numx = sum(1 if c == X else 0 for c in env.board)
        numo = sum(1 if c == O else 0 for c in env.board)
        assert numx - numo == 1 or numx - numo == 0
        env.current_player = O if numx > numo else X
        return env

    def copy(self):
        env = TttEnv()
        env.board = self.board[:]
        env.current_player = self.current_player
        return env

    def valid_actions(self):
        yield from valid_actions(self.board)

    def status(self):
        return status(self.board)

    def str1d(self):
        return "".join(["." if c == 0 else _CHAR_LOOKUP[c] for c in self.board])

    def str1d_sep(self, sep: str):
        b = self.str1d()
        return f"{b[:3]}{sep}{b[3:6]}{sep}{b[6:]}"

    def str2d(self):
        b = self.str1d()
        return f"{b[:3]}\n{b[3:6]}\n{b[6:]}"

    def step(self, action) -> tuple[Board, int, bool, bool, None]:
        """Reward assumes player/agent is X
        Returns board, reward, game_over?, False, None"""
        self._step_count += 1
        if self.board[action] != EMPTY:  # invalid action loses game
            return self.board, -1, True, False, None
        self.board[action] = self.current_player
        self.current_player = X if self.current_player == O else O
        s = IN_PROGRESS if self._step_count < 5 else status(self.board)
        reward = -1 if s == O else 1 if s == X else 0
        done = s != IN_PROGRESS
        return self.board, reward, done, False, None

    def __repr__(self):
        return self.str1d_sep("|") + " " + ("X" if self.current_player == X else "O")


class GymEnv(gym.Env):
    def __init__(self):
        self._env = TttEnv()
        self.action_space = _ACTION_SPACE
        self.observation_space = _OBS_SPACE

    def reset(self, seed=None, options=None):
        self._env = TttEnv()
        return self._obs(), {}

    @property
    def board(self):
        return self._env.board

    @staticmethod
    def from_str(s: str):
        env = GymEnv()
        env._env = TttEnv.from_str(s)
        return env

    def copy(self):
        env = GymEnv()
        env.board = self.board[:]
        env.current_player = self.current_player
        return env

    def valid_actions(self):
        yield from self._env.valid_actions()

    def step(self, action) -> tuple[np.ndarray, int, bool, bool, dict]:
        _, reward, term, trunc, _ = self._env.step(action)
        return self._obs(), reward, term, trunc, {}

    def status(self):
        return self._env.status()

    def str1d(self):
        return self._env.str1d()

    def str2d(self):
        return self._str("\n")

    def __repr__(self):
        return self.str1d()

    def _str(self, sep: str):
        b = "".join("x" if c == X else "o" if c == O else "." for c in self.board)
        return f"{b[:3]}{sep}{b[3:6]}{sep}{b[6:]}"

    def _obs(self):
        return np.array(self._env.board).reshape((3, 3))


class EnvWithOpponent(GymEnv):
    def __init__(self, opponent):
        super().__init__()
        self.opponent = opponent

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term or trunc:
            return obs, reward, term, trunc, info
        op_action = self.opponent.get_action(self)
        return super().step(op_action)


class IllegalActionError(Exception):
    pass


_winning_combinations = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def board2str1d(board: Board, sep=None):
    b = "".join(["." if c == 0 else _CHAR_LOOKUP[c] for c in board])
    return b if not sep else f"{b[:3]}{sep}{b[3:6]}{sep}{b[6:]}"


def board2str2d(board: Board):
    b = board2str1d(board)
    return f"{b[:3]}\n{b[3:6]}\n{b[6:]}"


def str2board(s: str):
    """returns board, current_player"""
    board = [EMPTY] * 9
    for i, c in enumerate(s.replace("|", "").lower()):
        if c == "x":
            board[i] = X
        elif c == "o":
            board[i] = O
        elif c == ".":
            board[i] = EMPTY
        else:
            raise ValueError(f"Invalid character in board string: {c}")
    numx = sum(1 if c == X else 0 for c in board)
    numo = sum(1 if c == O else 0 for c in board)
    assert numx - numo == 1 or numx - numo == 0
    current_player = O if numx > numo else X
    return board, current_player


def winner(board: Board):
    for a, b, c in _winning_combinations:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return None


def status(board) -> Status:
    w = winner(board)
    if w is not None:
        return t.cast(Status, w)
    if any(x == EMPTY for x in board):
        return t.cast(Status, IN_PROGRESS)
    return t.cast(Status, DRAW)


def valid_actions(board: Board):
    for i in range(9):
        if board[i] == EMPTY:
            yield i


def other_player(player: Player):
    return X if player == O else O


def symmetrics(board: str):
    assert len(board) == 9
    nboard = np.array([ord(c) for c in board]).reshape((3, 3))
    symmetries: list[np.ndarray] = []

    # Identity
    symmetries.append(nboard)

    # Rotations
    for k in range(1, 4):
        symmetries.append(np.rot90(nboard, k))

    # Reflections
    symmetries.append(np.fliplr(nboard))  # Horizontal reflection
    symmetries.append(np.flipud(nboard))  # Vertical reflection
    symmetries.append(np.transpose(nboard))  # Diagonal (main)
    symmetries.append(np.fliplr(np.transpose(nboard)))  # Diagonal (anti)

    for s in symmetries:
        yield "".join(
            chr(i)
            for i in s.reshape(
                9,
            )
        )
