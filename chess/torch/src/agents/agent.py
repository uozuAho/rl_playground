from typing import Protocol

import chess
from env import env


class ChessAgent(Protocol):
    def get_action(self, game: env.ChessGame) -> chess.Move:
        pass

    def get_actions(self, games: list[env.ChessGame]) -> list[chess.Move]:
        pass
