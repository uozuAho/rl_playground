import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import ttt.env
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent
from ttt.env import TicTacToeEnv


N_ENVS = 16


class TrainedAgent:
    def __init__(self, model):
        self._model = model

    @staticmethod
    def from_name(name):
        model = DQN.load(name)
        return TrainedAgent(model)

    @staticmethod
    def from_model(model):
        return TrainedAgent(model)

    def get_action(self, env: TicTacToeEnv):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self._model.predict(obs, deterministic=True)
        return action.item()


class MyEvalCallback(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % (1000 * N_ENVS) == 0:
            print(f"train: {self.num_timesteps} steps")
            agent = TrainedAgent.from_model(self.model)
            opponent = RandomAgent()
            my_eval(agent, opponent, 100)
        return True


def make_env(opponent):
    return TicTacToeEnv(
        opponent=opponent,
        on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER
    )


def train(name, opponent, steps, callback=None):
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
        verbose=1
    )
    model.learn(
        total_timesteps=steps,
        log_interval=steps//30,
        callback=MyEvalCallback()
    )
    model.save(name)
    return TrainedAgent.from_model(model)


def my_eval(a, opponent, num_games=100):
    env = TicTacToeEnv(opponent=opponent)
    wins = 0
    losses = 0
    draws = 0
    illegal_actions = 0
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        done = False
        reward = 0
        while not done:
            action = a.get_action(env)
            try:
                obs, reward, term, trunc, info = env.step(action)
            except ttt.env.IllegalActionError:
                reward = -1
                term = True
                illegal_actions += 1
            done = term or trunc
            total_reward += reward
        wins += 1 if reward == 1 else 0
        losses += 1 if reward == -1 else 0
        draws += 1 if reward == 0 else 0
        avg_reward = total_reward / num_games
    print(f'eval: {num_games} games. wins: {wins}, draws: {draws}, losses: {losses}, ill moves: {illegal_actions}. Avg reward: {avg_reward}')


def eval_trained_ppo_agents(names):
    for name in names:
        agent = TrainedAgent.from_name(name)
        print(f"'{name}' vs random:")
        my_eval(agent, RandomAgent())
        print(f"'{name}' vs perfect:")
        my_eval(agent, PerfectAgent('O'))


agent = train('dqn-mlp-vs-rng', opponent=RandomAgent(), steps=200000)
# train('dqn-mlp-vs-perfect', opponent=PerfectAgent('O'), steps=100000)
# dspiel = train2('dqn-spiel-rng', 400000)
# eval_trained_ppo_agents(['dqn-mlp-vs-rng', 'dqn-mlp-vs-perfect', 'dqn-spiel-rng'])
my_eval(agent, RandomAgent())
