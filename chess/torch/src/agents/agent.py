from abc import ABC, abstractmethod
import random

import chess
from env import env


class ChessAgent(ABC):
    @abstractmethod
    def get_action(self, env: env.ChessGame) -> chess.Move:
        raise NotImplementedError()


class RandomAgent(ChessAgent):
    def __init__(self, player: env.Player):
        self.player = player

    def get_action(self, env):
        assert env.turn == self.player
        return random.choice(list(env.legal_moves()))
