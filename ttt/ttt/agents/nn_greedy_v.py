from collections import deque
from dataclasses import dataclass
import random
import typing as t

import torch
import torch.nn as nn
import torch.optim as optim

from ttt.agents.agent import TttAgent
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


class NnGreedyVAgent(TttAgent):
    """
    NN greedy value learning agent. Doesn't improve with training. Dunno why.
    """
    def __init__(self, device):
        self.nn = LinearFC()
        self.device = device

    def get_action(self, env: TicTacToeEnv):
        return self._greedy_action(env)

    def _greedy_action(self, env: TicTacToeEnv):
        max_move = None
        max_val = -9999999999999999
        for m in env.valid_actions():
            temp_env = env.copy()
            temp_env.step(m)
            val = self._nn_out(temp_env.board)
            if val > max_val:
                max_move = m
                max_val = val
        return max_move

    def _e_greedy_action(self, env: TicTacToeEnv, epsilon: float):
        if random.uniform(0, 1.0) < epsilon:
            return random.choice(list(env.valid_actions()))
        else:
            return self._greedy_action(env)

    def state_val(self, env: TicTacToeEnv):
        return self._nn_out(env.board)

    def action_values(self, board_str: str):
        """ For debugging """
        vals = {}
        env = TicTacToeEnv.from_str(board_str)
        for a in env.valid_actions():
            temp = env.copy()
            temp.step(a)
            vals[a] = self.state_val(temp).item()
        return vals

    def _nn_out(self, state: BoardState) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.nn(state_t)

    def train(self,
            env: TicTacToeEnv,
            n_training_episodes,
            min_epsilon=0.001,
            max_epsilon=1.0,
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=10,
            replay_buffer_size=32,
            n_ep_update_interval=10,
            ep_callback: t.Optional[t.Callable[[int, float], None]]=None
            ):
        print(f"training for {n_training_episodes} episodes...")
        buffer = ReplayBuffer(replay_buffer_size)
        optimiser = optim.Adam(self.nn.parameters(), lr=learning_rate)
        e = epsilon.exp_decay_gen(max_epsilon, min_epsilon, n_training_episodes)
        for i in range(n_training_episodes):
            env.reset()
            done = False
            eps = e.__next__()
            while not done:
                state = env.board[:]
                if env.current_player == 'X':
                    action = self._e_greedy_action(env, eps)
                else:
                    action = random.choice(list(env.valid_actions()))
                _, reward, done, _, _ = env.step(action)
                buffer.add(GameStep(state, env.board[:], reward, done))
            if len(buffer) > batch_size and i % n_ep_update_interval == 0:
                optimise_net(
                    self.nn,
                    buffer.sample(batch_size),
                    optimiser,
                    self.device,
                    gamma)
            if ep_callback:
                ep_callback(i, 0.0)  # bogus e value, don't care
        print('training done')


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
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


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
        q_next[non_final_mask] = value_net(non_final_next_states).squeeze(1)
    q_next = (rewards + gamma * q_next).unsqueeze(1)

    q = value_net(states)

    optimiser.zero_grad()
    criterion = nn.SmoothL1Loss()
    loss = criterion(q, q_next)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
    optimiser.step()

    # learn final values
    if any(b.is_end for b in batch):
        final_states = torch.stack(
            [torch.tensor(b.next_state, dtype=torch.float32) for b in batch if b.is_end]).to(device)
        rewards = torch.tensor([b.reward for b in batch if b.is_end], device=device).unsqueeze(1)

        q = value_net(final_states)

        optimiser.zero_grad()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q, rewards)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        optimiser.step()
