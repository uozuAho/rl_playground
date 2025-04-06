import random
from ttt.agents.agent import TttAgent2
import ttt.env2 as env2


class RandomAgent(TttAgent2):
    def get_action(self, env: env2.Env):
        assert isinstance(env, env2.Env)
        return random.choice(list(env.valid_actions()))
