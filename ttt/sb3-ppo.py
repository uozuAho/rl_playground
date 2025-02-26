import numpy as np
from stable_baselines3 import PPO
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


def show_env_params():
    env = make_env(RandomAgent())

    print("action space")
    print(env.action_space)
    print(env.action_space.n)

    print("observation space")
    print(env.observation_space)


def run_env_demo():
    env = make_env(opponent=RandomAgent())
    observation, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample(mask=env.action_masks())
        print("Action taken:", action)

        observation, reward, done, _, info = env.step(action)
        env.render()

    env.close()


def train_ppo_agent(name, opponent, steps):
    env = make_vec_env(lambda: make_env(opponent=opponent), n_envs=16)

    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 128,
        n_epochs = 4,
        learning_rate=0.1,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        device='cpu',
        verbose=1)
    model.learn(total_timesteps=steps)
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
        self.model = PPO.load(name, device='cpu')

    def get_action(self, env: TicTacToeEnv):
        # hack env internals to get obs
        obs = np.array(env.board).reshape((3,3))
        action, _ = self.model.predict(obs)
        return action.item()


def eval_trained_ppo_agents(names):
    for name in names:
        print(f"'{name}' vs random:")
        my_eval(TrainedAgent(name), RandomAgent())
        print(f"'{name}' vs perfect:")
        my_eval(TrainedAgent(name), PerfectAgent('O'))


def run_trained_ppo_agent(name):
    env = make_env(opponent=RandomAgent())
    observation, info = env.reset()

    model = PPO.load(name)

    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


# show_env_params()
# run_env_demo()
train_ppo_agent('ppo-mlp-vs-rng', opponent=RandomAgent(), steps=100000)
# train_ppo_agent('ppo-mlp-vs-perfect', opponent=PerfectAgent('O'), steps=100000)
# eval_hard_coded_agents()
# run_trained_ppo_agent()
eval_trained_ppo_agents(['ppo-mlp-vs-rng'])
# eval_trained_ppo_agents(['ppo-mlp-vs-rng', 'ppo-mlp-vs-perfect'])
