import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from ttt.agents.agent import TttAgent
from ttt.agents.compare import play_and_report2
from ttt.agents.random import RandomAgent
import ttt.env


N_ENVS = 16


class Sb3DqnAgent(TttAgent):
    def __init__(self, model: DQN):
        self._model = model

    @staticmethod
    def load(name_or_path):
        model = DQN.load(name_or_path)
        return Sb3DqnAgent(model)

    @staticmethod
    def from_model(model):
        return Sb3DqnAgent(model)

    def get_action(self, env: ttt.env.Env):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self._model.predict(obs, deterministic=True)
        return action.item()

    def save(self, path):
        self._model.save(path)

    @staticmethod
    def train_new(opponent, steps, verbose: bool):
        env = make_vec_env(lambda: make_env(opponent=opponent), n_envs=N_ENVS)

        # using params found with optuna
        model = DQN(
            policy='MlpPolicy',
            batch_size=200,
            buffer_size=4000,
            gamma=.98,
            learning_rate=0.003,
            learning_starts=1500,
            target_update_interval=1000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.08,
            env = env,
            policy_kwargs={
                "net_arch": [32, 32]
            },
            verbose=1 if verbose else 0
        )
        model.learn(
            total_timesteps=steps,
            log_interval=steps//30,
            callback=MyEvalCallback() if verbose else None
        )
        return Sb3DqnAgent.from_model(model)


class MyEvalCallback(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % (1000 * N_ENVS) == 0:
            print(f"train: {self.num_timesteps} steps")
            agent = Sb3DqnAgent.from_model(self.model)
            opponent = RandomAgent()
            play_and_report2(agent, 'sb3dqn', opponent, 'rng', 100)
        return True


def make_env(opponent):
    return ttt.env.EnvWithOpponent(
        opponent=opponent,
        on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER
    )
