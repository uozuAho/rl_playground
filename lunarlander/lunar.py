import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def run_env_demo():
    """ Demonstrates the environment by making random actions """
    env = gym.make("LunarLander-v3", render_mode="human")
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
    """ When done, saves the model to a zip file in this dir """
    env = make_vec_env('LunarLander-v3', n_envs=16)
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
    model_name = "ppo-LunarLander-v3"
    model.save(model_name)


def eval_agent():
    model_name = "ppo-LunarLander-v3"
    model = PPO.load(model_name, device='cpu')
    eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def run_trained_agent():
    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset()

    model_name = "ppo-LunarLander-v3"
    model = PPO.load(model_name, device='cpu')

    done = False
    while not done:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


run_env_demo()
# train_agent(1000)
# eval_agent()
# run_trained_agent()