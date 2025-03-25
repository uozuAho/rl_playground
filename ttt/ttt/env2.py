"""
Improvement over first env:

- simpler code
- faster
"""

import typing as t

EMPTY = 0
X = 1
O = -1
DRAW = 2
IN_PROGRESS = 3
type Status = t.Literal[-1, 1, 2, 3]
type Board = list[int]


class Env:
    """ Assumes player/agent is X """
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_player = X
        self.board = [EMPTY] * 9

    def valid_actions(self):
        for i in range(9):
            if self.board[i] == EMPTY:
                yield i

    def step(self, action):
        self.board[action] = self.current_player
        self.current_player = X if self.current_player == O else O
        s = status(self.board)
        reward = -1 if s == O else 1 if s == X else 0
        done = s != IN_PROGRESS
        return self.board, reward, done, False, None


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
    if w is not None: return w
    if any(x == EMPTY for x in board):
        return IN_PROGRESS
    return DRAW

