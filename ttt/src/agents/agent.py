from abc import ABC
from pathlib import Path
import ttt.env as t3


class TttAgent(ABC):
    def get_action(self, env: t3.TttEnv) -> int:
        raise NotImplementedError()

    def get_actions(self, envs: list[t3.TttEnv]) -> list[int]:
        raise NotImplementedError()

    def save(self, path: str | Path):
        return NotImplementedError()
