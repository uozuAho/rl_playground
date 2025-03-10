from ttt.agents.agent import TttAgent
from ttt.env import TicTacToeEnv


class RandomAgent(TttAgent):
    def get_action(self, env: TicTacToeEnv):
        return env.action_space.sample(mask=env.valid_action_mask())
