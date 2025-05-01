""" Greedy v learner. Learns board values then plays greedily on moves that
    result in the highest board value.

    Doesn't do very well as x after 5000 eps of training. Maybe NN arch isn't
    great. Can play as o.

    Todo later
    - add double learning
    - perf (maybe) learn final board values as part of the main batch
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim

from ttt.agents.agent import TttAgent
import ttt.env as t3
import utils.epsilon


type Player = t.Literal['O', 'X']
type GameStatus = t.Literal['O', 'X', 'draw', 'in_progress']



@dataclass
class GameStep:
    state: t3.Board
    next_state: t3.Board
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
        states = torch.stack([self.state2input(b.state) for b in batch]).to(self.device)
        rewards = torch.tensor([b.reward for b in batch], device=self.device)
        non_final_mask = torch.tensor([not b.is_end for b in batch], device=self.device)
        non_final_next_states = torch.stack(
            [self.state2input(b.next_state) for b in batch if not b.is_end]).to(self.device)
        q_next = torch.zeros(len(batch), device=self.device)
        with torch.no_grad():
            # todo: double learning: use target net here
            q_next[non_final_mask] = self(non_final_next_states).squeeze(1)
        q_next = (rewards + self.gamma * q_next).unsqueeze(1)

        q = self(states)

        self.optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q, q_next)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimiser.step()

        # learn final board values
        if any(b.is_end for b in batch):
            final_states = torch.stack(
                [self.state2input(b.next_state) for b in batch if b.is_end]).to(self.device)
            rewards = torch.tensor([b.reward for b in batch if b.is_end], device=self.device).unsqueeze(0).t()

            q = self(final_states)

            self.optimiser.zero_grad()
            criterion = nn.SmoothL1Loss()
            loss = criterion(q, rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimiser.step()

    def state2input(self, state: t3.Board):
        # reshape board list to 2d, unsqueeze to add channels for conv2d
        return torch.tensor(state, dtype=torch.float32).reshape((3,3)).unsqueeze(0)


class NnGreedyVAgent(TttAgent):
    def __init__(self, device: str):
        self.nn = ConvNet(lr=1e-4, gamma=0.9, device=device).to(device)
        self.device = device

    def get_action(self, env: t3.Env, epsilon=0.0):
        if random.random() < epsilon:
            return random.choice(list(env.valid_actions()))
        return self._greedy_action(env)

    def save(self, path: str | Path):
        torch.save(self.nn.state_dict(), path)

    @staticmethod
    def load(path: Path | str, device: str):
        agent = NnGreedyVAgent(device)
        agent.nn.load_state_dict(torch.load(path, weights_only=True))
        return agent

    @staticmethod
    def train_new(opponent: TttAgent, n_eps: int, device: str):
        agent = NnGreedyVAgent(device)
        agent.train(opponent, n_eps)
        return agent

    def state_val(self, env: t3.Env):
        return self._nn_out(env.board)

    def board_val(self, board: t3.Board):
        return self._nn_out(board)

    def train(
            self,
            opponent: TttAgent,
            n_episodes: int,
            epsilon: t.Optional[t.Iterator[float]] = None,
            buffer_size = 64,
            batch_size = 64,
            n_ep_update_interval = 5,
            callback = None
            ):
        print(f"training for {n_episodes} episodes...")
        buffer = ReplayBuffer(buffer_size)
        epsilon = epsilon or utils.epsilon.exp_decay_gen(0.5, 0.01, n_episodes)
        for i in range(n_episodes):
            eps = epsilon.__next__()
            for step in play_game(self, opponent, eps):
                buffer.add(step)
            if len(buffer) > batch_size and i % n_ep_update_interval == 0:
                self.nn.learn_batch(buffer.sample(batch_size))
            if callback:
                callback(i)
        print('training done')

    def action_vals(self, env: t3.Env):
        """ For debugging """
        vals = {}
        for a in env.valid_actions():
            temp = env.copy()
            temp.step(a)
            vals[str(temp)] = self.state_val(temp).item()
        return vals

    def _greedy_action(self, env: t3.Env):
        def next_state(board, action, player):
            next_board = board[:]
            next_board[action] = player
            return next_board

        actions = list(env.valid_actions())
        next_states = [next_state(env.board, a, env.current_player) for a in actions]
        if env.current_player == t3.X:
            action_idx = self._nn_out_batch(next_states).argmax().item()
        else:
            # assumes trained playing as x. best o move is worst x move ... right?
            action_idx = self._nn_out_batch(next_states).argmin().item()
        return actions[action_idx]

    def _nn_out(self, state: t3.Board) -> float:
        # unsqueeze to batch of 1
        state_t = self.nn.state2input(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.nn(state_t).item()

    def _nn_out_batch(self, states: list[t3.Board]):
        states_t = torch.stack([self.nn.state2input(s) for s in states]).to(self.device)
        with torch.no_grad():
            return self.nn(states_t)


def play_game(agent_x: NnGreedyVAgent, opponent_o: TttAgent, epsilon: float):
    env = t3.FastEnv()
    done = False
    while not done:
        state = env.board[:]
        if env.current_player == t3.X:
            action = agent_x.get_action(env, epsilon)
        else:
            action = opponent_o.get_action(env)
        _, reward, done, _, _ = env.step(action)
        yield GameStep(state, env.board[:], reward, done)
