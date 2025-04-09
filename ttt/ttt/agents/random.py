import random
from ttt.agents.agent import TttAgent
from ttt.env import Env


class RandomAgent(TttAgent):
    def get_action(self, env: Env):
        return random.choice(list(env.valid_actions()))
