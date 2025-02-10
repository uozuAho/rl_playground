import numpy as np

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym


ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_env(show_graphics=False):
    r_mode = "human" if show_graphics else "rgb_array"
    return gym.make(ENV_NAME, render_mode=r_mode)


def show_env_params():
    env = make_env()

    print("action space")
    print(env.action_space)

    print("observation space")
    print(env.observation_space)


def run_env_demo():
    env = make_env(show_graphics=True)
    observation, _ = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        print("Action taken:", action)

        observation, reward, terminated, truncated, _ = env.step(action)
        print("Obs:", observation)

        done = terminated or truncated

    env.close()


class Policy(nn.Module):
    # h = hidden
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    @staticmethod
    def from_env(env: gym.Env, h_size=64):
        s_size = env.observation_space.shape[0]
        a_size = env.action_space.n
        return Policy(s_size, a_size, h_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    env = make_env()
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))

    return scores


def train(num_episodes=1000):
    print(f'training for {num_episodes} episodes...')
    hidden_layer_size = 16
    learning_rate = 1e-2
    env = make_env()
    policy = Policy.from_env(env, hidden_layer_size).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    reinforce(
        policy,
        optimizer,
        n_training_episodes=num_episodes,
        max_t=1000,
        gamma=1.0,
        print_every=100)
    return policy


def evaluate_agent(policy: Policy, max_steps=1000, n_eval_episodes=10):
    env = make_env()
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, term, trunc, info = env.step(action)
            done = term or trunc
            total_rewards_ep += reward
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def run_trained_agent(policy: Policy):
    env = make_env(show_graphics=True)
    observation, _ = env.reset()

    done = False
    while not done:
        action, _ = policy.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


# show_env_params()
# run_env_demo()
policy = train(300)
# evaluate_agent(policy)
run_trained_agent(policy)
