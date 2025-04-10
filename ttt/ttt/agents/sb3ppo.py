import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import ttt.env


class Sb3PpoAgent:
    def __init__(self, model: PPO):
        self.model = model

    def get_action(self, env: ttt.env.Env):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self.model.predict(obs)
        return action.item()

    def save(self, name_or_path):
        self.model.save(name_or_path)

    @staticmethod
    def load(name_or_path):
        model = PPO.load(name_or_path, device='cpu')
        return Sb3PpoAgent(model)

    @staticmethod
    def train_new(opponent, steps, verbose=False):
        env = make_vec_env(lambda: make_env(opponent=opponent), n_envs=16)
        model = PPO(
            policy = 'MlpPolicy',
            env = env,
            n_steps = 2048,
            batch_size = 128,
            n_epochs = 4,
            learning_rate=0.1,
            gamma = 0.999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            device='cpu',
            verbose=1 if verbose else 0)
        model.learn(total_timesteps=steps)
        return Sb3PpoAgent(model)


def make_env(opponent):
    return ttt.env.EnvWithOpponent(
        opponent=opponent,
        invalid_action_response=ttt.env.INVALID_ACTION_GAME_OVER
    )
