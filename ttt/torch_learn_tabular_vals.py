""" Grasping at straws here. My NN greedy V agent isn't learning, dunno why.
    Can I make an NN approximate the tabular agent's table?
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

from ttt.agents.tab_greedy_v import GreedyVAgent


class Simple(nn.Module):
    """ Can't learn more than about 10 state values with accuracy """
    def __init__(self):
        super(Simple, self).__init__()
        self.l1 = nn.Linear(9, 32, dtype=torch.float32)
        self.l2 = nn.Linear(32, 32, dtype=torch.float32)
        self.l3 = nn.Linear(32, 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


class SimpleConv(nn.Module):
    # Based on https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L122C5-L133C29
    # loss: avg: 0.081, max: 0.645
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.activation = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, state_str: str):
        x = self._state2input(state_str)
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def _state2input(self, state_str: str):
        nums = state2nums(state_str)
        i = torch.tensor(nums, dtype=torch.float32).reshape((3,3))
        i = i.unsqueeze(0) # channels = 1
        i = i.unsqueeze(0) # batch of 1
        return i


def state2nums(state_str: str):
    state_str = state_str.replace('|', '')
    assert len(state_str) == 9
    return [2 if c == 'x' else 1 if c == 'o' else 0 for c in state_str]


def state2tensor1d(state_str: str):
    nums = state2nums(state_str)
    return torch.tensor(nums, dtype=torch.float32)


net = SimpleConv()
optimiser = optim.Adam(net.parameters(), lr=1e-4)
tab_agent = GreedyVAgent.load('tabular-greedy-v.json')
q_table = tab_agent._q_table
all_vals = list(q_table.values())[:500]


for i in range(99999999999999999):
    random.shuffle(all_vals)
    losses = []
    for state, value in all_vals:
        q_est = net(state)
        q_actual = torch.tensor([value])

        optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    print(f'loss: avg: {sum(losses)/len(losses):.3f}, max: {max(losses):.3f}')
