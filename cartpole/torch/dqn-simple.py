""" A simplified version of dqn.py

    No enhancements such as replay memory, double learning, gradient clipping.
    Simply updates the model after each environment step.

    Learns much more slowly than dqn.py
"""

from collections import namedtuple
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4

env = gym.make("CartPole-v1")
device = 'cpu'
print(f'Using device: {device}')
n_actions = env.action_space.n   # 2 actions: left, right
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
steps_done = 0


def select_action(state, deterministic=False):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if not deterministic:
        steps_done += 1
    if random.random() > eps_threshold or deterministic:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def plot_durations(episode_durations):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Episode duration (s) vs training episode')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.show()


def optimize_model(transition: Transition):
    # Compute Q(s_t, a)
    action_values = policy_net(transition.state)  # Q(s_t)
    action = transition.action.item()             # a
    state_action_value = action_values[:, action] # Q(s_t, a)

    # Compute V(s_{t+1}) = reward + GAMMA(argmax_a(Q(s_t+1)))
    # todo: this should be with torch.no_grad:?
    next_state_value = torch.zeros(1) if transition.next_state is None else \
                       policy_net(transition.next_state).max(1).values

    expected_state_action_value = transition.reward + GAMMA * next_state_value

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_value, expected_state_action_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train(num_episodes):
    """ Returns list(episode duration) """
    episode_durations = []

    print(f'training for {num_episodes} episodes...')

    for e in range(num_episodes):
        if e % 10 == 0:
            print(f'episode {e}/{num_episodes}')
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            optimize_model(Transition(state, action, next_state, reward))

            state = next_state

            if done:
                episode_durations.append(t + 1)
                break

    return episode_durations


def evaluate(n_eps):
    print('evaluating...')
    episode_rewards = []
    for _ in range(n_eps):
        state, _ = env.reset()
        done = False
        total_rewards_ep = 0

        for step in range(200):
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state, deterministic=True)
            new_state, reward, term, trunc, info = env.step(action.item())
            done = term or trunc
            total_rewards_ep += reward
            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = sum(episode_rewards) / len(episode_rewards)

    print(f'avg reward: {mean_reward}')


evaluate(20)
durations = train(1024)
evaluate(20)
plot_durations(durations)
