import random

import env.connect4 as c4
from agents.agent import Agent


class RandomAgent(Agent):
    def get_action(self, state: c4.GameState):
        return random.choice(c4.get_valid_moves(state))

    def get_actions(self, states: list[c4.GameState]):
        return [self.get_action(s) for s in states]
