import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

import ttt.env
from ttt.agents.perfect import PerfectAgent
from ttt.agents.random import RandomAgent
from ttt.env import TicTacToeEnv



def make_env(opponent):
    return TicTacToeEnv(
        opponent=opponent,
        on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER
    )


def train(name, opponent, steps):
    env = make_vec_env(lambda: make_env(opponent=opponent), n_envs=16)

    model = DQN(
        policy = 'MlpPolicy',
        batch_size=1024,
        env = env,
        verbose=1
    )
    model.learn(total_timesteps=steps, log_interval=steps//10)
    model.save(name)


def my_eval(a, opponent, num_games=50):
    env = TicTacToeEnv(opponent=opponent)
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    for _ in range(num_games):
        env.reset()
        done = False
        reward = 0
        while not done:
            action = a.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_reward += reward
        wins += 1 if reward == 1 else 0
        losses += 1 if reward == -1 else 0
        draws += 1 if reward == 0 else 0
        avg_reward = total_reward / num_games
    print(f'{num_games} games. wins: {wins}, draws: {draws}, losses: {losses}. Avg reward: {avg_reward}')


class TrainedAgent:
    def __init__(self, name):
        self.path = name
        self.model = DQN.load(name)

    def get_action(self, env: TicTacToeEnv):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self.model.predict(obs, deterministic=True)
        return action.item()


def eval_trained_ppo_agents(names):
    for name in names:
        print(f"'{name}' vs random:")
        my_eval(TrainedAgent(name), RandomAgent())
        print(f"'{name}' vs perfect:")
        my_eval(TrainedAgent(name), PerfectAgent('O'))


# train('dqn-mlp-vs-rng', opponent=RandomAgent(), steps=200000)
eval_trained_ppo_agents(['dqn-mlp-vs-rng'])
