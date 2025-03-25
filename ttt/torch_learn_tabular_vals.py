""" Grasping at straws here. My NN greedy V agent isn't learning, dunno why. Can
    I make an NN approximate the tabular agent's table?

    Yes. MidConv does better than the simple models. Dunno why such a complex
    module is needed when the sb3 dqn model can get good results with a fully
    connected 9-32-32-9 policy network.
"""

import random
import time
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from ttt.agents.tab_greedy_v import GreedyVAgent


class Simple(nn.Module):
    """ Can't learn more than about 10 state values with accuracy """
    # loss: avg: 0.081, max: 0.748
    def __init__(self):
        super(Simple, self).__init__()
        self.l1 = nn.Linear(9, 32, dtype=torch.float32)
        self.l2 = nn.Linear(32, 32, dtype=torch.float32)
        self.l3 = nn.Linear(32, 1, dtype=torch.float32)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state_str: str):
        x = self._state2input(state_str)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def learn_single(self, state: str, value: float):
        q_est = net(state)
        q_actual = torch.tensor([value])

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state_str: str):
        return torch.tensor(state2nums(state_str), dtype=torch.float32)


class SimpleConv(nn.Module):
    # Based on https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L122C5-L133C29
    # similar perf to simple
    # loss: avg: 0.081, max: 0.645
    def __init__(self):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.activation = nn.Sigmoid()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 1)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, state_str: str):
        x = self._state2input(state_str)
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def learn_single(self, state: str, value: float):
        q_est = net(state)
        q_actual = torch.tensor([value]).unsqueeze(0)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state_str: str):
        nums = state2nums(state_str)
        i = torch.tensor(nums, dtype=torch.float32).reshape((3,3))
        i = i.unsqueeze(0) # channels = 1
        i = i.unsqueeze(0) # batch of 1
        return i


class MidConv(nn.Module):
    # Does better! Was still learning when I stopped it:
    # loss: avg: 0.016, max: 0.032 after 1 minute, batch size 50
    def __init__(self, device):
        super(MidConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def learn_batch(self, batch: t.List[t.Tuple[str, float]]):
        states = [b[0] for b in batch]
        vals = [b[1] for b in batch]
        state_batch = self._states2batch(states)
        q_est = net(state_batch)
        q_actual = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).t().to(device)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state: str):
        nums = state2nums(state)
        i = torch.tensor(nums, dtype=torch.float32, device=self.device).reshape((3,3))
        i = i.unsqueeze(0) # channels = 1
        i = i.unsqueeze(0) # batch of 1
        return i

    def _states2batch(self, states: t.List[str]):
        nums = [state2nums(s) for s in states]
        # unsqueeze -> 1 channel for conv2d
        ts = [torch.tensor(n, dtype=torch.float32).reshape((3, 3)).unsqueeze(0) for n in nums]
        batch = torch.stack(ts).to(self.device)
        return batch



def state2nums(state_str: str):
    state_str = state_str.replace('|', '')
    assert len(state_str) == 9
    return [2 if c == 'x' else 1 if c == 'o' else 0 for c in state_str]


def batches(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
# device = 'cpu'
print(f'using device {device}')

batch_size = 50
net = MidConv(device).to(device)
torchinfo.summary(net, input_size=(batch_size, 1, 3, 3))
tab_agent = GreedyVAgent.load('tabular-greedy-v.json')
q_table = tab_agent._q_table
all_vals = list(q_table.values())

start = time.time()
t_prev = time.time()
for i in range(99999999999999999):
    random.shuffle(all_vals)
    losses = []

    # todo: support learn_batch in all models
    # update net for every state,value:
    # for state, value in all_vals:
    #     loss = net.learn_single(state, value)
    #     losses.append(loss)

    # update net in batches
    for batch in batches(all_vals, batch_size):
        loss = net.learn_batch(batch)
        losses.append(loss)

    tt = time.time() - start
    tn = time.time() - t_prev
    t_prev = time.time()
    avg_loss = sum(losses)/len(losses)
    print(f'{tt:3.1f} ({tn:.2f}): loss: avg: {avg_loss:.3f}, max: {max(losses):.3f}')
