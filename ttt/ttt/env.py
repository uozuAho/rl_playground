# Originally copied from https://github.com/haje01/gym-tictactoe/blob/master/gym_tictactoe/env.py
# Modified until I could get it working with stable baselines

import typing as t
import numpy as np
import gymnasium as gym
from gymnasium import spaces


NUM_LOC = 9
IN_PROGRESS_REWARD = 0
WIN_REWARD = 1
LOSS_REWARD = -1
INVALID_ACTION_REWARD = -1
EMPTY_CODE = 0
O_CODE = 1
X_CODE = 2

IN_PROGRESS = -1
DRAW = 0
O_WIN = 1
X_WIN = 2

# invalid action responses
INVALID_ACTION_THROW = 0
INVALID_ACTION_GAME_OVER = 1


def tomark(code):
    if code == O_CODE: return 'O'
    elif code == X_CODE: return 'X'
    else: return ' '


def tocode(mark):
    if mark == 'O': return O_CODE
    if mark == 'X': return X_CODE
    if mark == ' ': return EMPTY_CODE
    raise Exception(f"Invalid mark: '{mark}'")


def next_mark(mark):
    return 'X' if mark == 'O' else 'O'


def check_game_status(board):
    """Return game status by current board status.

    Args:
        board (list): Current board state

    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [O_CODE, X_CODE]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            return IN_PROGRESS

    return DRAW


class IllegalActionError(Exception):
    pass


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 my_mark='X',
                 opponent=None,
                 on_invalid_action=INVALID_ACTION_THROW):
        """
        Params
            - opponent: has a get_action(env) method
        """
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Box(low=0, high=2, shape=(3,3), dtype=np.int8)
        self.my_mark = my_mark
        self.opponent = opponent
        self.on_invalid_action = on_invalid_action
        self.reset()

    @staticmethod
    def from_str(board_str: str):
        board_str = board_str.replace('|', '')
        assert len(board_str) == 9
        assert board_str.upper().count('X') >= board_str.upper().count('O')
        env = TicTacToeEnv()
        env.board = [tocode(c.upper()) for c in board_str]
        if board_str.upper().count('X') > board_str.upper().count('O'):
            env.next_mark = 'O'
        status = check_game_status(env.board)
        if status >= 0:
            env.is_game_over = True
        return env

    def __str__(self):
        b = ''.join(tomark(x) for x in self.board)
        return f'{b[:3]}|{b[3:6]}|{b[6:]}'

    def reset(self, **kwargs):
        self.board = [0] * NUM_LOC
        self.next_mark = 'X'  # X always goes first
        self.is_game_over = False
        return self._get_obs(), {}

    def copy(self):
        c = TicTacToeEnv()
        c.my_mark = self.my_mark
        c.next_mark = self.next_mark
        c.opponent = self.opponent
        c.board = self.board[:]
        c.is_game_over = self.is_game_over
        c.on_invalid_action = self.on_invalid_action
        return c

    @property
    def current_player(self) -> t.Literal['X', 'O']:
        """ Alias for next_mark """
        return self.next_mark

    def get_status(self):
        """ Returns one of

            - IN_PROGRESS
            - DRAW
            - O_WIN
            - X_WIN
        """
        return check_game_status(self.board)

    def step(self, action):
        """ Make a move, then if there's an opponent, make their move.
            Returns:
            state, reward, game_over?, truncated?, info
        """
        if not self.opponent:
            return self._step(action)
        else:
            obs, reward, done, _, _ = self._step(action)
            if done:
                return obs, reward, done, False, {}
            op_action = self.opponent.get_action(self)
            return self._step(op_action)

    def _step(self, action):
        """ Make a move, then hand over to the opponent """
        assert self.action_space.contains(action)

        if self.is_game_over:
            return self._get_obs(), 0, True, False, {}

        loc = action
        if self.board[loc] != 0:
            if self.on_invalid_action == INVALID_ACTION_GAME_OVER:
                self.is_game_over = True
                return self._get_obs(), INVALID_ACTION_REWARD, True, False, {}
            elif self.on_invalid_action == INVALID_ACTION_THROW:
                raise IllegalActionError(f"action: {loc}: position already filled")
            else:
                raise Exception(f"unknown invalid action response: {self.on_invalid_action}")

        reward = IN_PROGRESS_REWARD
        self.board[loc] = tocode(self.next_mark)
        status = check_game_status(self.board)
        if status >= 0:
            self.is_game_over = True
            if status in [1, 2]:
                reward = WIN_REWARD if status == tocode(self.my_mark) else LOSS_REWARD

        self.next_mark = next_mark(self.next_mark)
        return self._get_obs(), reward, self.is_game_over, False, {}

    def valid_action_mask(self):
        return np.array(
            [0 if x else 1 for x in self.board],
            dtype=np.int8
        )

    def valid_actions(self):
        for a, is_valid in enumerate(self.valid_action_mask()):
            if is_valid:
                yield a

    def _get_obs(self):
        return np.array(self.board).reshape((3,3))

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)  # NOQA
            print('')
        else:
            self._show_board(print)

    def show_episode(self, human, episode):
        self._show_episode(print, episode)

    def _show_episode(self, showfn, episode):
        showfn("==== Episode {} ====".format(episode))

    def _show_board(self, showfn):
        """Draw tictactoe board."""
        for j in range(0, 9, 3):
            def mark(i):
                return tomark(self.board[i]) if True or\
                    self.board[i] != 0 else str(i+1)
            showfn('  ' + '|'.join([mark(i) for i in range(j, j+3)]))
            if j < 6:
                showfn('  ' + '-----')

    def show_turn(self, human, mark):
        self._show_turn(print, mark)

    def _show_turn(self, showfn, mark):
        showfn("{}'s turn.".format(mark))

    def show_result(self, human, mark, reward):
        self._show_result(print, mark, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn('')


if __name__ == "__main__":
    env = TicTacToeEnv()
    obs = env.reset()
    print("reset. obs:", obs)

    terminated = False
    while not terminated:
        # action = env.action_space.sample()
        action = env.action_space.sample(mask=env.valid_action_mask())
        print('action:', action)

        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward)

        env.render()

    env.show_result(True, 'not used', 'not used')
