from collections import deque
from dataclasses import dataclass
import random
import time
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
import utils.epsilon as epsilon


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


class ConvNet(nn.Module):
    def __init__(self, gamma, lr, device):
        super(ConvNet, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 1)
        self.gamma = gamma
        self.optimiser = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def learn_batch(self, batch: t.List[GameStep]):
        """SARSA update:
        Q(s,a) <- Q(s,a) + lr*(R + gamma * Q(s_t+1,a_t+1) - Q(s,a))

        We're only estimating state values, so:

        Q(s) <- Q(s) + lr*(R + gamma * Q(s_t+1) - Q(s))
        """
        states = torch.stack([self.state2input(b.state) for b in batch]).to(device)
        rewards = torch.tensor([b.reward for b in batch], device=device)
        non_final_mask = torch.tensor([not b.is_end for b in batch], device=device)
        non_final_next_states = torch.stack(
            [self.state2input(b.next_state) for b in batch if not b.is_end]).to(device)
        q_next = torch.zeros(len(batch), device=device)
        with torch.no_grad():
            # todo: double learning: use target net here
            q_next[non_final_mask] = value_net(non_final_next_states).squeeze(1)
        q_next = (rewards + self.gamma * q_next).unsqueeze(1)

        q = value_net(states)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q, q_next)
        loss.backward()
        self.optimiser.step()

        return loss.item()

    def state2input(self, state: BoardState):
        # reshape board list to 2d, add channels for conv2d
        return torch.tensor(state, dtype=torch.float32).reshape((3,3)).unsqueeze(0)


class GreedyTdAgent(TttAgent):
    def __init__(self, nn: ConvNet, device: str):
        self.nn = nn
        self.device = device

    def get_action(self, env: TicTacToeEnv, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(list(env.valid_actions()))
        max_move = None
        max_val = -999
        for m in env.valid_actions():
            temp_env = env.copy()
            temp_env.step(m)
            val = self._nn_out(temp_env.board)
            if val > max_val:
                max_move = m
                max_val = val
        return max_move

    def state_val(self, env: TicTacToeEnv):
        return self._nn_out(env.board)

    def action_vals(self, env: TicTacToeEnv):
        """ For debugging """
        vals = {}
        for a in env.valid_actions():
            temp = env.copy()
            temp.step(a)
            vals[str(temp)] = self.state_val(temp, learn=False).item()
        return vals

    def _nn_out(self, state: BoardState) -> float:
        # unsqueeze to batch of 1
        state_t = self.nn.state2input(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.nn(state_t)


def gamestatus(env: TicTacToeEnv) -> GameStatus:
    state = env.get_status()
    if state == ttt.env.O_WIN: return 'O'
    if state == ttt.env.X_WIN: return 'X'
    if state == ttt.env.DRAW: return 'draw'
    return 'in_progress'


def play_game(agent_x: GreedyTdAgent, opponent_o: TttAgent, epsilon: float):
    env = TicTacToeEnv()
    assert env.my_mark == 'X'
    done = False
    while not done:
        state = env.board[:]
        if env.current_player == 'X':
            action = agent_x.get_action(env, epsilon)
        else:
            action = opponent_o.get_action(env)
        _, reward, done, _, _ = env.step(action)
        yield GameStep(state, env.board[:], reward, done)


def train(
        agent_x: GreedyTdAgent,
        opponent_o: TttAgent,
        n_episodes: int,
        epsilon: t.Iterator[float],
        buffer_size = 64,
        batch_size = 64,
        n_ep_update_interval = 5,
        ):
    print(f"training for {n_episodes} episodes...")
    buffer = ReplayBuffer(buffer_size)
    for i in range(n_episodes):
        eps = epsilon.__next__()
        for step in play_game(agent_x, opponent_o, eps):
            buffer.add(step)
        if len(buffer) > batch_size and i % n_ep_update_interval == 0:
            agent_x.nn.learn_batch(buffer.sample(batch_size))
    print('training done')


def eval_agent(agent: GreedyTdAgent, opponent: TttAgent):
    play_and_report(agent, "greedyTd", opponent, "rando?", 100)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = 'cpu'
print(f'using device {device}')

value_net = ConvNet(lr=1e-4, gamma=0.9, device=device).to(device)
agent = GreedyTdAgent(value_net, device)
opponent = RandomAgent()
eval_agent(agent, opponent)
print("PRESS CTRL-C TO STOP!")
start = time.time()
prev = time.time()
for _ in range(1000):
    print(f'{time.time() - start:3.1f}s ({time.time() - prev:.2f}s)')
    prev = time.time()
    eps = epsilon.exp_decay_gen(0.3, 0.001, 1000)
    train(
        agent,
        opponent,
        n_episodes=1000,
        buffer_size=32,
        batch_size=16,
        n_ep_update_interval=4,
        epsilon=eps)
    eval_agent(agent, opponent)
