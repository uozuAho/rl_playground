from abc import ABC
import ttt.env2 as ttt2


class TttAgent2(ABC):
    def get_action(self, env: ttt2.Env):
        return NotImplementedError()
