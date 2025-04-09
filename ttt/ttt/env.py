import typing as t


EMPTY = 0
X = 1
O = -1  # noqa: E741
DRAW = 2
IN_PROGRESS = 3
INVALID_ACTION_THROW = 0
INVALID_ACTION_GAME_OVER = 1
type Action = int
type Player = t.Literal[-1, 1]
type Status = t.Literal[-1, 1, 2, 3]
type Board = list[int]


class Env:
    def __init__(self, invalid_action_response=INVALID_ACTION_THROW):
        self.reset()
        self.invalid_action_response = invalid_action_response

    def reset(self):
        self.current_player = X
        self.board = [EMPTY] * 9

    @staticmethod
    def from_str(str):
        env = Env()
        for i, c in enumerate(str.replace('|', '').lower()):
            if c == 'x':
                env.board[i] = X
            elif c == 'o':
                env.board[i] = O
            elif c == '.':
                env.board[i] = EMPTY
            else:
                raise ValueError(f'Invalid character in board string: {c}')
        numx = sum(1 if X else 0 for c in env.board)
        numo = sum(1 if O else 0 for c in env.board)
        assert numx - numo == 1 or numx - numo == 0
        env.current_player = O if numx > numo else X
        return env

    def copy(self):
        env = Env()
        env.board = self.board[:]
        env.current_player = self.current_player
        return env

    def valid_actions(self):
        yield from valid_actions(self.board)

    def step(self, action) -> tuple[Board, int, bool, bool, dict]:
        """ Reward assumes player/agent is X """
        if self.board[action] != EMPTY:
            if self.invalid_action_response == INVALID_ACTION_GAME_OVER:
                return self.board, -1, True, False, {}
            else:
                raise IllegalActionError()
        do_action(self.current_player, action, self.board)
        self.current_player = other_player(self.current_player)
        s = status(self.board)
        reward = -1 if s == O else 1 if s == X else 0
        done = s != IN_PROGRESS
        return self.board, reward, done, False, {}

    def status(self):
        return status(self.board)

    def winner(self):
        return winner(self.board)


class EnvWithOpponent(Env):
    def __init__(self, opponent, invalid_action_response=INVALID_ACTION_THROW):
        super().__init__(invalid_action_response)
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


def do_action(player: Player, action: Action, board: Board):
    board[action] = player


def other_player(player: Player):
    return X if player == O else O
