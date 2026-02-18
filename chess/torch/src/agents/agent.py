from typing import Protocol

import chess
from env import env


class ChessAgent(Protocol):
    def get_action(self, game: env.ChessGame) -> chess.Move:
        pass
