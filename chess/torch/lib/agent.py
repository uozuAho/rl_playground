from abc import ABC, abstractmethod

import chess
from lib import env


class ChessAgent(ABC):
    @abstractmethod
    def get_action(self, env: env.ChessGame) -> chess.Move:
        raise NotImplementedError()
