import random

from agents.agent import ChessAgent
from env import env


class RandomAgent(ChessAgent):
    def get_action(self, game: env.ChessGame):
        return random.choice(list(game.legal_moves()))

    def get_actions(self, games: list[env.ChessGame]):
        return [self.get_action(g) for g in games]
