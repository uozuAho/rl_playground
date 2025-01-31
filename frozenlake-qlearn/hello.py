import random
import gymnasium as gym
import numpy as np


def demo_env():
    env = gym.make("FrozenLake-v1",
                   map_name="4x4", is_slippery=False, render_mode="human")
    observation, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        print("Action taken:", action)

        observation, reward, terminated, truncated, info = env.step(action)
        print("Obs:", observation)

        done = terminated or truncated

    env.close()


def greedy_policy(qtable: np.ndarray, state):
    action = np.argmax(qtable[state][:])
    return action


def egreedy_policy(env, qtable, state, epsilon):
    random_num = random.uniform(0, 1)
    if random_num > epsilon:
        action = greedy_policy(qtable, state)
    else:
        action = env.action_space.sample()

    return action


def train(
        n_training_episodes,
        ep_max_steps,
        min_epsilon,    # epsilon: exploration rate
        max_epsilon,
        eps_decay_rate,
        learning_rate,
        gamma,          # discount rate
        env,
        Qtable,
        ):
    for episode in range(n_training_episodes):
        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) *
            np.exp(-eps_decay_rate * episode))
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False

        for step in range(ep_max_steps):
            action = egreedy_policy(env, Qtable, state, epsilon)

            new_state, reward, terminated, truncated, info = env.step(action)

            Qtable[state][action] = Qtable[state][action] + learning_rate * (
                reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action]
            )

            if terminated or truncated:
                break

            state = new_state
    return Qtable


def train_agent():
    env = gym.make("FrozenLake-v1",
                   map_name="4x4", is_slippery=False, render_mode="rgb_array")

    state_space = env.observation_space.n
    action_space = env.action_space.n
    qtable = np.zeros((state_space, action_space))

    train(n_training_episodes=10000,
          ep_max_steps=99,
          learning_rate=0.7,
          min_epsilon=.05,
          max_epsilon=1.0,
          eps_decay_rate=.0005,
          gamma=.95,
          env=env,
          Qtable=qtable)

    np.save("qtable.npy", qtable)


def evaluate_agent(max_steps=99, n_eval_episodes=10):
    env = gym.make("FrozenLake-v1",
                   map_name="4x4", is_slippery=False, render_mode="rgb_array")
    qtable = np.load("qtable.npy")

    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action = greedy_policy(qtable, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


def run_trained_agent():
    env = gym.make("FrozenLake-v1",
                   map_name="4x4", is_slippery=False, render_mode="human")
    observation, info = env.reset()
    qtable = np.load("qtable.npy")

    done = False
    while not done:
        action = greedy_policy(qtable, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


demo_env()
# train_agent()
# evaluate_agent()
# run_trained_agent()
