""" Same as greed-v, but adds a replay buffer and TD learning from replay
    samples. Also broken. Doesn't learn/improve.
"""

from collections import deque
from dataclasses import dataclass
import random
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ttt.agents.agent import TttAgent
from ttt.agents.compare import play_and_report
from ttt.agents.random import RandomAgent
import ttt.env
from ttt.env import TicTacToeEnv


type Player = t.Literal['O', 'X']
type GameStatus = t.Literal['O', 'X', 'draw', 'in_progress']
type BoardState = list[int]


@dataclass
class GameStep:
    state: BoardState
    next_state: BoardState
    reward: int
    is_end: bool


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def add(self, step: GameStep):
        self.memory.append(step)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LinearFC(nn.Module):
    """ Fully connected linear/sequential NN """
    def __init__(self):
        super(LinearFC, self).__init__()
        self.l1 = nn.Linear(9, 32, dtype=torch.float32)
        self.l2 = nn.Linear(32, 32, dtype=torch.float32)
        self.l3 = nn.Linear(32, 1, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class GreedyTdAgent(TttAgent):
    def __init__(self, nn: LinearFC, device: str):
        self.nn = nn
        self.device = device

    def get_action(self, env: TicTacToeEnv, learn=False):
        max_move = None
        max_val = -999
        for m in env.valid_actions():
            temp_env = env.copy()
            temp_env.step(m)
            val = self._nn_out(temp_env.board, learn)
            if val > max_val:
                max_move = m
                max_val = val
        return max_move

    def state_val(self, env: TicTacToeEnv, learn=False):
        return self._nn_out(env.board, learn)

    def action_vals(self, env: TicTacToeEnv):
        """ For debugging """
        vals = {}
        for a in env.valid_actions():
            temp = env.copy()
            temp.step(a)
            vals[str(temp)] = self.state_val(temp, learn=False).item()
        return vals

    def _nn_out(self, state: BoardState, learn=False) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if learn:
            return self.nn(state_t)
        else:
            with torch.no_grad():
                return self.nn(state_t)


def gamestatus(env: TicTacToeEnv) -> GameStatus:
    state = env.get_status()
    if state == ttt.env.O_WIN: return 'O'
    if state == ttt.env.X_WIN: return 'X'
    if state == ttt.env.DRAW: return 'draw'
    return 'in_progress'


def optimise_net(
        value_net: LinearFC,
        batch: list[GameStep],
        optimiser: optim.Optimizer,
        device: str,
        gamma=0.9):
    """SARSA update:
    Q(s,a) <- Q(s,a) + lr*(R + gamma * Q(s_t+1,a_t+1) - Q(s,a))

    We're only estimating state values, so:

    Q(s) <- Q(s) + lr*(R + gamma * Q(s_t+1) - Q(s))
    """
    states = torch.stack([torch.tensor(b.state, dtype=torch.float32) for b in batch]).to(device)
    rewards = torch.tensor([b.reward for b in batch], device=device)
    non_final_mask = torch.tensor([not b.is_end for b in batch], device=device)
    non_final_next_states = torch.stack(
        [torch.tensor(b.next_state, dtype=torch.float32) for b in batch if not b.is_end]).to(device)
    q_next = torch.zeros(len(batch), device=device)
    with torch.no_grad():
        # todo: double learning: use target net here
        q_next[non_final_mask] = value_net(non_final_next_states).squeeze(1)
    q_next = (rewards + gamma * q_next).unsqueeze(1)

    q = value_net(states)

    optimiser.zero_grad()
    criterion = nn.MSELoss()
    loss = criterion(q, q_next)
    loss.backward()
    optimiser.step()


def play_game(agent_x: GreedyTdAgent, opponent_o: TttAgent):
    env = TicTacToeEnv()
    assert env.my_mark == 'X'
    done = False
    while not done:
        state = env.board[:]
        if env.current_player == 'X':
            action = agent_x.get_action(env, learn=False)
        else:
            action = opponent_o.get_action(env)
        _, reward, done, _, _ = env.step(action)
        yield GameStep(state, env.board[:], reward, done)


def train(
        agent_x: GreedyTdAgent,
        opponent_o: TttAgent,
        n_episodes: int,
        device: str,
        batch_size = 64,
        n_ep_update_interval = 5,
        ):
    print(f"training for {n_episodes} episodes...")
    buffer = ReplayBuffer(1000)
    optimiser = optim.SGD(agent_x.nn.parameters(), lr=1e-4)
    for i in range(n_episodes):
        for step in play_game(agent_x, opponent_o):
            buffer.add(step)
        if len(buffer) > batch_size and i % n_ep_update_interval == 0:
            optimise_net(agent_x.nn, buffer.sample(batch_size), optimiser, device)
    print('training done')

def eval_agent(agent: GreedyTdAgent, opponent: TttAgent):
    play_and_report(agent, "greedyTd", opponent, "rando?", 100)


def demo_nn_learn():
    """ I'm struggling to understand why the greedy agent isn't learning.
        This works: Learning random output values for random inputs with LinearFC
    """
    net = LinearFC()
    def randstate():
        return [random.randint(0, 2) for _ in range(9)]
    states = [randstate() for _ in range(10)]
    vals = [random.random() for _ in range(10)]
    states_t = [torch.tensor(x, dtype=torch.float32) for x in states]
    vals_t = [torch.tensor(x) for x in vals]
    optimiser = optim.SGD(net.parameters(), lr=.01)
    for i in range(1000):
        losses = []
        for j in range(len(states)):
            idx = i % len(states)
            input = states_t[idx]
            exp_out = vals_t[idx]
            nn_out = net(input)

            optimiser.zero_grad()
            criterion = nn.MSELoss()
            loss = criterion(nn_out, exp_out)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        a = avg_loss


def demo_nn_learn_batch():
    """ Same as above, but learn/output in batches"""
    net = LinearFC()
    def randstate():
        return [random.randint(0, 2) for _ in range(9)]
    states = [randstate() for _ in range(10)]
    vals = [random.random() for _ in range(10)]
    states_t = torch.stack([torch.tensor(x, dtype=torch.float32) for x in states])
    vals_t = torch.tensor(vals).unsqueeze(1)
    optimiser = optim.SGD(net.parameters(), lr=.01)
    for i in range(1000):
        input = states_t
        exp_out = vals_t
        nn_out = net(input)

        optimiser.zero_grad()
        criterion = nn.MSELoss()
        loss = criterion(nn_out, exp_out)
        loss.backward()
        optimiser.step()
        print(loss.item())


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')

# demo_nn_learn()
# demo_nn_learn_batch()

value_net = LinearFC().to(device)
agent = GreedyTdAgent(value_net, device)
opponent = RandomAgent()
eval_agent(agent, opponent)
print("PRESS CTRL-C TO STOP!")
for _ in range(1000):
    train(agent, opponent, 1000, device, batch_size=64)
    eval_agent(agent, opponent)
