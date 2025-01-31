import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_NAME = "Blackjack-v1"


def show_env_params():
    env = gym.make(ENV_NAME, render_mode="human")

    print("action space")
    print(env.action_space)

    print("observation space")
    print(env.observation_space)


def run_env_demo():
    env = gym.make(ENV_NAME, render_mode="human")
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
    env = make_vec_env(ENV_NAME, n_envs=16)
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
    model_name = "ppo-blackjack"
    model.save(model_name)


# show_env_params()
# run_env_demo()
train_agent(1000)
