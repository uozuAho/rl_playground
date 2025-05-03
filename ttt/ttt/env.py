from __future__ import annotations
from abc import ABC, abstractmethod
import typing as t
import gymnasium as gym
from gymnasium import spaces
import numpy as np


EMPTY = 0
X = 1
O = -1  # noqa: E741
DRAW = 2
IN_PROGRESS = 3
type Action = int
type Player = t.Literal[-1, 1]
type Status = t.Literal[-1, 1, 2, 3]
type Board = list[int]


_ACTION_SPACE = spaces.Discrete(9)
_OBS_SPACE = spaces.Box(low=-1, high=1, shape=(3,3), dtype=np.int8)
_CHAR_LOOKUP = ['x', 'o']


ObsType = t.TypeVar('ObsType')


class Env(ABC, t.Generic[ObsType]):
    def __init__(self):
        self.board = []
        self.current_player = X

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> Env:
        raise NotImplementedError()

    @abstractmethod
    def status(self) -> Status:
        raise NotImplementedError()

    @abstractmethod
    def str1d(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def valid_actions(self) -> t.Iterable[int]:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action) -> tuple[ObsType, int, bool, bool, t.Any]:
        raise NotImplementedError()


class FastEnv(Env[Board]):
    """ Minimal, going for all out speed. TODO use this for Env """
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_player = X
        self.board = [EMPTY] * 9
        self._step_count = 0

    @staticmethod
    def from_str(str: str):
        env = FastEnv()
        for i, c in enumerate(str.replace('|', '').lower()):
            if c == 'x':
                env.board[i] = X
            elif c == 'o':
                env.board[i] = O
            elif c == '.':
                env.board[i] = EMPTY
            else:
                raise ValueError(f'Invalid character in board string: {c}')
        numx = sum(1 if c == X else 0 for c in env.board)
        numo = sum(1 if c == O else 0 for c in env.board)
        assert numx - numo == 1 or numx - numo == 0
        env.current_player = O if numx > numo else X
        return env

    def copy(self):
        env = FastEnv()
        env.board = self.board[:]
        env.current_player = self.current_player
        return env

    def valid_actions(self):
        yield from valid_actions(self.board)

    def status(self):
        return status(self.board)

    def str1d(self):
        return ''.join(['.' if c == 0 else _CHAR_LOOKUP[c] for c in self.board])

    def str2d(self):
        b = self.str1d()
        return f'{b[:3]}\n{b[3:6]}\n{b[6:]}'

    def step(self, action) -> tuple[Board, int, bool, bool, None]:
        """ Reward assumes player/agent is X """
        self._step_count += 1
        if self.board[action] != EMPTY:  # invalid action loses game
            return self.board, -1, True, False, None
        self.board[action] = self.current_player
        self.current_player = X if self.current_player == O else O
        s = IN_PROGRESS if self._step_count < 5 else status(self.board)
        reward = -1 if s == O else 1 if s == X else 0
        done = s != IN_PROGRESS
        return self.board, reward, done, False, None


class GymEnv(gym.Env, Env[np.ndarray]):
    def __init__(self):
        self._env = FastEnv()
        self.action_space = _ACTION_SPACE
        self.observation_space = _OBS_SPACE

    def reset(self, seed=None, options=None):
        self._env = FastEnv()
        return self._obs(), {}

    @property
    def board(self):
        return self._env.board

    @staticmethod
    def from_str(str):
        env = GymEnv()
        env._env = FastEnv.from_str(str)
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
        return self._str('\n')

    def __repr__(self):
        return self.str1d()

    def _str(self, sep: str):
        b = ''.join('x' if c == X else 'o' if c == O else '.' for c in self.board)
        return f'{b[:3]}{sep}{b[3:6]}{sep}{b[6:]}'

    def _obs(self):
        return np.array(self._env.board).reshape((3,3))


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
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def winner(board: Board):
    for a, b, c in _winning_combinations:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]


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
