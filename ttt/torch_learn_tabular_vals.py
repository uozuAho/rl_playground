""" Grasping at straws here. My NN greedy V agent isn't learning, dunno why.
    Can I make an NN approximate the tabular agent's table?
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

from ttt.agents.tab_greedy_v import GreedyVAgent


class LinearFC(nn.Module):
    def __init__(self):
        super(LinearFC, self).__init__()
        self.l1 = nn.Linear(9, 32, dtype=torch.float32)
        self.l2 = nn.Linear(32, 32, dtype=torch.float32)
        self.l3 = nn.Linear(32, 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def state2tensor(state_str: str):
    state_str = state_str.replace('|', '')
    assert len(state_str) == 9
    nums = [2 if c == 'x' else 1 if c == 'o' else 0 for c in state_str]
    return torch.tensor(nums, dtype=torch.float32)


net = LinearFC()
optimiser = optim.Adam(net.parameters(), lr=1e-4)
tab_agent = GreedyVAgent.load('tabular-greedy-v.json')
q_table = tab_agent._q_table
all_vals = list(q_table.values())[:500]


for i in range(99999999999999999):
    random.shuffle(all_vals)
    losses = []
    for state, value in all_vals:
        input = state2tensor(state)
        q_est = net(input)
        q_actual = torch.tensor([value])

        optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    print(f'loss: avg: {sum(losses)/len(losses):.3f}, max: {max(losses):.3f}')
