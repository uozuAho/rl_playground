""" Initial code from pytorch tute:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Features:
- double learning: https://medium.com/@helena.godart/double-deep-q-networks-an-introductory-guide-9b0d88310197
- minibatch with random sampling
- gradient clipping: https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem
"""

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
TAU = 0.005
LR = 1e-4

env = gym.make("CartPole-v1")
device = 'cpu'
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )
print(f'Using device: {device}')
n_actions = env.action_space.n   # 2 actions: left, right
state, info = env.reset()
n_observations = len(state)

# double learning: policy & target net
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0


def select_action(state, deterministic=False):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold or deterministic:
        with torch.no_grad():
            # Return the index of the max output value.
            # This will either be 0 or 1, since there are 2 neurons in the
            # final layer. These correspond to the estimated action value for
            # the given input state, ie. the output is [value(left), value(right)]
            # max: https://pytorch.org/docs/stable/generated/torch.max.html#torch.max
            # view: https://pytorch.org/docs/stable/tensor_view.html#tensor-view-doc
            # max(1).indices.view(1,1) = 1x1 view of the index of the max value in column 1
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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # transpose list(Transition) to Transition(state=list, action=list, ...)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute batch Q(s_t, a)
    #
    # policy_net(states) = estimated action values for each given state
    # policy_net(states) = [
    #                       [value(left), value(right)],
    #                       [value(left), value(right)],
    #                       ...
    #                      ]
    #
    # action_batch = action taken at each corresponding state, eg. [1, 0, 0, 1 , ...]
    #
    # tensor.gather selects the policy action value by the index in action_batch
    # eg. for action values [1, 0] the output would be [value(right), value(left)]
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute batch V(s_{t+1})
    #
    # Only non-final states are used. V = 0 for final states.
    #
    # q-learning:      the .max(1).values selects the highest value action (this
    #                  is what differentiates q-learning from sarsa)
    # double learning: target_net is used to estimate the expected value
    # no_grad:         this is not a learning step for target_net, no_grad
    #                  instructs torch to not save the gradient info for the
    #                  given input
    non_final_mask = torch.tensor(list(s is not None for s in batch.next_state),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    # loss function = difference between actual and expected output. In generic
    # q learning, the loss is (reward + gamma * max_next_value - value)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()  # zero out accumulated gradients: https://stackoverflow.com/a/48009142
    loss.backward()        # backpropagate loss (calculated gradients). This
                           # works since state_action_values is a torch tensor
                           # that came from the policy network, thus is has
                           # access to gradient info and the network computation
                           # graph

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()       # updates the policy network weights using gradients


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

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            # double learning: Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

    return episode_durations


def evaluate(n_eps):
    print('evaluating...')
    episode_rewards = []
    for episode in range(n_eps):
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
durations = train(256)
evaluate(20)
plot_durations(durations)
