from abc import ABC
from ttt.env import TicTacToeEnv
import ttt.env2 as ttt2


class TttAgent(ABC):
    def get_action(self, env: TicTacToeEnv):
        return NotImplementedError()



class TttAgent2(ABC):
    def get_action(self, env: ttt2.Env):
        return NotImplementedError()
