from typing import Protocol
import env.connect4 as c4

type Action = int  # column in which to place a piece


class Agent(Protocol):
    def get_actions(self, states: list[c4.GameState]) -> list[Action]:
        pass
