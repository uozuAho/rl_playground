import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from ttt.agents.agent import TttAgent
import ttt.env


class Sb3MaskPpoAgent(TttAgent):
    def __init__(self, model: MaskablePPO):
        self.model = model

    def get_action(self, env: ttt.env.Env):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self.model.predict(obs, action_masks=mask_fn(env))
        return action.item()

    def save(self, name_or_path):
        self.model.save(name_or_path)

    @staticmethod
    def load(name_or_path):
        model = MaskablePPO.load(name_or_path)
        return Sb3MaskPpoAgent(model)

    def train_new(opponent, steps, verbose=False):
        env = make_vec_env(lambda: make_env(opponent=opponent), n_envs=16)
        model = MaskablePPO(
            policy = 'MlpPolicy',
            env = env,
            n_steps = 2048,
            batch_size = 128,
            n_epochs = 4,
            learning_rate=0.1,
            gamma = 0.999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            verbose=1 if verbose else 0)
        model.learn(total_timesteps=steps)
        return Sb3MaskPpoAgent(model)


def make_env(opponent):
    return ActionMasker(ttt.env.EnvWithOpponent(opponent=opponent), mask_fn)


def mask_fn(env: ttt.env.Env):
    valid_actions = list(env.valid_actions())
    return [a in valid_actions for a in range(9)]
