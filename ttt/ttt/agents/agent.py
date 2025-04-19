from abc import ABC
from pathlib import Path
import ttt.env as t3


class TttAgent(ABC):
    def get_action(self, env: t3.Env):
        return NotImplementedError()

    def save(self, path: str | Path):
        return NotImplementedError()
