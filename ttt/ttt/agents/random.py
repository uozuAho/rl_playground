import random
from ttt.agents.agent import TttAgent
import ttt.env as t3


class RandomAgent(TttAgent):
    def get_action(self, env: t3.Env):
        return random.choice(list(env.valid_actions()))
