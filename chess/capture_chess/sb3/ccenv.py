import chess
import gymnasium as gym
from gymnasium import spaces
from RLC.capture_chess.environment import Board


class CaptureChess(gym.Env):
    """ Gym wrapper around capture chess environment """
    def __init__(self):
        self._board = Board()
        self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,8,8))

    def reset(self):
        self._board.reset()
        return self._board.layer_board, {}

    def step(self, action: int):
        from_square = action // 64
        to_square = action % 64
        move = chess.Move(from_square, to_square)
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
