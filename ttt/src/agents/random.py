import random
from agents.agent import TttAgent
import ttt.env as t3


class RandomAgent(TttAgent):
    def get_action(self, env: t3.TttEnv):
        return random.choice(list(env.valid_actions()))

    def get_actions(self, envs: list[t3.TttEnv]) -> list[int]:
        return [self.get_action(e) for e in envs]
