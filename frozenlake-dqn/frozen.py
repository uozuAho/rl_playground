import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_NAME = "FrozenLake-v1"
MAP_NAME = "4x4"
IS_SLIPPERY = False
SAVED_MODEL_NAME = "ppo-frozenlake"


def make_env(show_graphics=True):
    return gym.make(
        ENV_NAME,
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode="human" if show_graphics else None
    )

def show_env_params():
    env = make_env()

    print("action space")
    print(env.action_space)

    print("observation space")
    print(env.observation_space)


def run_env_demo():
    env = make_env()
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
    env = make_vec_env(lambda: make_env(show_graphics=False), n_envs=16)
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


def eval_agent():
    model = PPO.load(SAVED_MODEL_NAME, device='cpu')
    eval_env = Monitor(make_env(show_graphics=False))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def run_trained_agent():
    env = make_env(show_graphics=True)
    observation, info = env.reset()

    model = PPO.load(SAVED_MODEL_NAME, device='cpu')

    done = False
    while not done:
        action, _ = model.predict(observation)
        action = action.item()  # action is ndarray, env doesn't like that. boo
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


# show_env_params()
# run_env_demo()
# train_agent(100000)
# eval_agent()
run_trained_agent()
