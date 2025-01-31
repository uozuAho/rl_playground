import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def run_env_demo():
    env = gym.make("Blackjack-v1", render_mode="human")
    observation, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        print("Action taken:", action)

        observation, reward, terminated, truncated, info = env.step(action)
        print("Obs:", observation)

        done = terminated or truncated

    env.close()


run_env_demo()
