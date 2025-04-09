import random
from ttt.agents.agent import TttAgent2
from ttt.env import Env


class RandomAgent(TttAgent2):
    def get_action(self, env: Env):
        return random.choice(list(env.valid_actions()))
