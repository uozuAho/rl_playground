import random
from ttt.agents.agent import TttAgent, TttAgent2
from ttt.env import TicTacToeEnv
import ttt.env2 as env2


class RandomAgent(TttAgent):
    def get_action(self, env: TicTacToeEnv):
        return env.action_space.sample(mask=env.valid_action_mask())


class RandomAgent2(TttAgent2):
    def get_action(self, env: env2.Env):
        return random.choice(list(env.valid_actions()))
