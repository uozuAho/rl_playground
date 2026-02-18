from abc import ABC, abstractmethod

import chess
from env import env


class ChessAgent(ABC):
    @abstractmethod
    def get_action(self, game: env.ChessGame) -> chess.Move:
        raise NotImplementedError()
