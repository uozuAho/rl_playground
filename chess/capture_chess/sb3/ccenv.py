import chess
import gymnasium as gym
from gymnasium import spaces
from RLC.capture_chess.environment import Board


ILLEGAL_ACTION_THROW = 0
ILLEGAL_ACTION_NEG_REWARD = 1
ILLEGAL_ACTION_NEG_REWARD_GAME_OVER = 2


class IllegalActionError(Exception):
    def __init__(self, move: chess.Move):
        super().__init__()
        self.move = move


class CaptureChess(gym.Env):
    """ Gym wrapper around capture chess environment """
    def __init__(self, illegal_action_behaviour):
        self._board = Board()
        self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,8,8))
        self.illegal_action_behaviour = illegal_action_behaviour

    def reset(self, **kwargs):
        self._board.reset()
        return self._board.layer_board, {}

    def step(self, action: int):
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
        if not self._board.board.is_legal(move):
            if self.illegal_action_behaviour == ILLEGAL_ACTION_THROW:
                raise IllegalActionError(move)
            if self.illegal_action_behaviour == ILLEGAL_ACTION_NEG_REWARD:
                return self._board.layer_board, -100, False, False, {}
            if self.illegal_action_behaviour == ILLEGAL_ACTION_NEG_REWARD_GAME_OVER:
                return self._board.layer_board, -100, True, False, {}
        done, reward = self._board.step(move)
        # hack fix pawn promotion reward
        # reward should only be 1,3,5,9
        if reward % 2 == 0:
            reward = 0
        return self._board.layer_board, reward, done, False, {}

    def render(self):
        print(self._board.board)

    def legal_actions(self):
        for move in self._board.board.generate_legal_moves():
            yield move.from_square * 64 + move.to_square

    def current_obs(self):
        return self._board.layer_board
