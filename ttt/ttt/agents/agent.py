from abc import ABC
from pathlib import Path
import ttt.env as ttt2


class TttAgent(ABC):
    def get_action(self, env: ttt2.Env):
        return NotImplementedError()

    def save(self, path: str | Path):
        return NotImplementedError()
