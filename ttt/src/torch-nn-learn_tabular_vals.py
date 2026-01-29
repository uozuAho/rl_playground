"""Keep this, as it may be handy to build better NN architectures later.

My NN greedy V agent wasn't learning, couldn't figure out why. Can I make an
NN approximate the tabular agent's table?

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

from agents.tab_greedy_v import TabGreedyVAgent


class Simple(nn.Module):
    """Can't learn more than about 10 state values with accuracy"""

    # loss: avg: 0.081, max: 0.748
    def __init__(self, device):
        super(Simple, self).__init__()
        self.device = device
        self.l1 = nn.Linear(9, 32, dtype=torch.float32)
        self.l2 = nn.Linear(32, 32, dtype=torch.float32)
        self.l3 = nn.Linear(32, 1, dtype=torch.float32)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def print_summary(self, batch_size):
        torchinfo.summary(self, input_size=(batch_size, 9))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    def learn_batch(self, batch: t.List[t.Tuple[str, float]]):
        states = [b[0] for b in batch]
        vals = [b[1] for b in batch]
        state_batch = self._states2batch(states)
        q_est = self(state_batch)
        q_actual = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).t().to(DEVICE)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state_str: str):
        return torch.tensor(state2nums(state_str), dtype=torch.float32)

    def _states2batch(self, states: t.List[str]):
        batch = torch.stack([self._state2input(s) for s in states]).to(self.device)
        return batch


class Simple2(nn.Module):
    # 10 values: loss: avg: 0.000, max: 0.000
    # 100 values: 34.9s (0.01): loss: avg: 0.000, max: 0.002
    # 1000 values: 38.1s (0.02): loss: avg: 0.060, max: 0.079
    def __init__(self, device):
        super(Simple2, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(9, 27), nn.ReLU(), nn.Linear(27, 9), nn.ReLU(), nn.Linear(9, 1)
        )
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        return self.model(x)

    def print_summary(self, batch_size):
        torchinfo.summary(self, input_size=(batch_size, 9))

    def learn_batch(self, batch: t.List[t.Tuple[str, float]]):
        states = [b[0] for b in batch]
        vals = [b[1] for b in batch]
        state_batch = self._states2batch(states)
        q_est = self(state_batch)
        q_actual = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).t().to(DEVICE)

        self.optimiser.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state_str: str):
        return torch.tensor(state2nums(state_str), dtype=torch.float32)

    def _states2batch(self, states: t.List[str]):
        batch = torch.stack([self._state2input(s) for s in states]).to(self.device)
        return batch


class SimpleConv(nn.Module):
    # Based on https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/real_chess/agent.py#L122C5-L133C29
    # similar perf to simple
    # loss: avg: 0.081, max: 0.645
    def __init__(self, device):
        super(SimpleConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0)
        self.activation = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 1)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def print_summary(self, batch_size):
        torchinfo.summary(self, input_size=(batch_size, 1, 3, 3))

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

    def learn_batch(self, batch: t.List[t.Tuple[str, float]]):
        states = [b[0] for b in batch]
        vals = [b[1] for b in batch]
        state_batch = self._states2batch(states)
        q_est = self(state_batch)
        q_actual = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).t().to(DEVICE)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state_str: str):
        nums = state2nums(state_str)
        i = torch.tensor(nums, dtype=torch.float32).reshape((3, 3))
        i = i.unsqueeze(0)  # channels = 1
        i = i.unsqueeze(0)  # batch of 1
        return i

    def _states2batch(self, states: t.List[str]):
        nums = [state2nums(s) for s in states]
        # unsqueeze -> 1 channel for conv2d
        ts = [
            torch.tensor(n, dtype=torch.float32).reshape((3, 3)).unsqueeze(0)
            for n in nums
        ]
        batch = torch.stack(ts).to(self.device)
        return batch


class MidConv(nn.Module):
    # Does better! Was still learning when I stopped it:
    # loss: avg: 0.016, max: 0.032 after 1 minute, batch size 50
    # 1000 values, batch size 100: 40.8s (0.02): loss: avg: 0.002, max: 0.002
    def __init__(self, device):
        super(MidConv, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=2, padding=1
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.optimiser = optim.Adam(self.parameters(), lr=1e-4)

    def print_summary(self, batch_size):
        torchinfo.summary(self, input_size=(batch_size, 1, 3, 3))

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
        q_est = self(state_batch)
        q_actual = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).t().to(DEVICE)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_est, q_actual)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def _state2input(self, state: str):
        nums = state2nums(state)
        i = torch.tensor(nums, dtype=torch.float32, device=self.device).reshape((3, 3))
        i = i.unsqueeze(0)  # channels = 1
        i = i.unsqueeze(0)  # batch of 1
        return i

    def _states2batch(self, states: t.List[str]):
        nums = [state2nums(s) for s in states]
        # unsqueeze -> 1 channel for conv2d
        ts = [
            torch.tensor(n, dtype=torch.float32).reshape((3, 3)).unsqueeze(0)
            for n in nums
        ]
        batch = torch.stack(ts).to(self.device)
        return batch


def state2nums(state_str: str):
    state_str = state_str.replace("|", "")
    assert len(state_str) == 9
    return [1 if c == "x" else -1 if c == "o" else 0 for c in state_str]


def batches(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = 'cpu'
print(f"using device {DEVICE}")

# TEST = True  # quickly test that all nets can train
TEST = False
NUM_VALS_TO_LEARN = 1000  # set to 999999999999999 for max
BATCH_SIZE = 100
nets: list[nn.Module] = [
    # Simple(DEVICE),
    # Simple2(DEVICE),
    # SimpleConv(DEVICE),
    MidConv(DEVICE).to(DEVICE)
]
tab_agent = TabGreedyVAgent.load("trained_models/tmcts_sym_100k_30")
q_table = tab_agent._q_table
all_vals = list(q_table.values())[:NUM_VALS_TO_LEARN]

for net in nets:
    net.print_summary(BATCH_SIZE)
    print(f"Fitting net to {len(all_vals)} values. Batch size: {BATCH_SIZE}.")
    input("press any key to start, ctrl+c to stop")
    start = time.time()
    t_prev = time.time()
    for i in range(1 if TEST else 99999999999999999):
        try:
            random.shuffle(all_vals)
            losses = []

            for batch in batches(all_vals, BATCH_SIZE):
                loss = net.learn_batch(batch)
                losses.append(loss)

            tt = time.time() - start
            tn = time.time() - t_prev
            t_prev = time.time()
            avg_loss = sum(losses) / len(losses)
            print(
                f"{tt:3.1f}s ({tn:.2f}): loss: avg: {avg_loss:.3f}, max: {max(losses):.3f}"
            )
        except KeyboardInterrupt:
            print()
            print(
                f"{NUM_VALS_TO_LEARN} values, batch size {BATCH_SIZE}: "
                + f"{tt:3.1f}s ({tn:.2f}): loss: avg: {avg_loss:.3f}, max: {max(losses):.3f}"
            )
            break
