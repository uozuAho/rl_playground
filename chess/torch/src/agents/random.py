import random

from agents.agent import ChessAgent
from env import env


class RandomAgent(ChessAgent):
    def __init__(self, player: env.Player):
        self.player = player

    def get_action(self, game: env.ChessGame):
        assert game.turn == self.player
        return random.choice(list(game.legal_moves()))

    def get_actions(self, games: list[env.ChessGame]):
        return [self.get_action(g) for g in games]
