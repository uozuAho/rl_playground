from abc import ABC
from ttt.env import TicTacToeEnv


class TttAgent(ABC):
    def get_action(self, env: TicTacToeEnv):
        return NotImplementedError()
