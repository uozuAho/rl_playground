import gymnasium as gym

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gymnasium import spaces


ENV_NAME = "Blackjack-v1"
SAVED_MODEL_NAME = "ppo-blackjack"


class MyWrapper(gym.ObservationWrapper):
    """ I dunno if this is valid, but seems OK to me.

        Problem: SB3 PPO doesn't support Tuple observation space

        This solution:
            Convert the obs space to multidiscrete, since a tuple of discretes
            seems equivalent. See
            - https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple
            - https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete
    """
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Tuple), "Assumes blackjack env has a Tuple observation space"
        assert len(env.observation_space) == 3
        assert all(isinstance(space, spaces.Discrete) for space in env.observation_space), "All spaces must be Discrete"

        self.observation_space = spaces.MultiDiscrete([space.n for space in env.observation_space])

    def observation(self, obs):
        return [obs[0], obs[1], obs[2]]


def make_env(show_graphics=False):
    return MyWrapper(gym.make(ENV_NAME, render_mode="human" if show_graphics else None))


def show_env_params():
    env = make_env()

    print("action space")
    print(env.action_space)

    print("observation space")
    print(env.observation_space)


def run_env_demo():
    env = make_env(show_graphics=True)
    observation, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        print("Action taken:", action)

        observation, reward, terminated, truncated, info = env.step(action)
        print("Obs:", observation)

        done = terminated or truncated

    env.close()


def train_agent(steps):
    """
    PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#ppo
        - "Proximal Policy Optimisation" learning algo
    policy networks: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#policy-networks
        - mlp:   general input types
        - cnn:   image input types
        - multi: mix of above
    """
    env = make_vec_env(lambda: make_env(), n_envs=16)
    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 4,
        gamma = 0.999,
        gae_lambda = 0.98,
        ent_coef = 0.01,
        device='cpu',
        verbose=1)
    model.learn(total_timesteps=steps)
    model.save(SAVED_MODEL_NAME)


class RandomAgent():
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def predict(self, obs, deterministic=False, *args, **kwargs):
        action = self.action_space.sample()
        action = np.array([action]) # dunno why i have to do this. SB3 expects an ndarray
        return action, None

    def learn(self, *args, **kwargs):
        pass


def eval_random_agent():
    eval_env = Monitor(make_env(show_graphics=False))
    model = RandomAgent(eval_env)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=50, deterministic=True)
    print("Random agent:")
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def eval_trained_agent():
    model = PPO.load(SAVED_MODEL_NAME, device='cpu')
    eval_env = Monitor(make_env(show_graphics=False))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=50, deterministic=True)
    print("Trained agent:")
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def run_trained_agent():
    env = make_env(show_graphics=True)
    observation, info = env.reset()

    model = PPO.load(SAVED_MODEL_NAME, device='cpu')

    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


# show_env_params()
# run_env_demo()
train_agent(1000)
eval_random_agent()
eval_trained_agent()
# run_trained_agent()
