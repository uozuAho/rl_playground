""" Simulating greedy RL chess bot, without the chess """


from abc import ABC, abstractmethod
import random
import collections
import time
from typing import Deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


WHITE = 1
BLACK = -1


def main():
    opponent = RandomAgent()

    # print("cpu:")
    # agent = GreedyChessAgent(device="cpu")
    # agent.train_against(opponent, 100)

    # print()

    print("gpu:")
    agent = GreedyChessAgent(device="cuda")
    agent.train_against(opponent, 2000)


class FakeChessGame:
    def __init__(self):
        self.turn = WHITE
        self._done = False
        self._done_prev = False
        self._winner = None
        self._winner_prev = None
        self._total_halfmoves = 0
        self._state = np.random.rand(8, 8, 8).astype(np.float32)
        self._state_prev = self._state

    def legal_moves(self):
        return [random.randint(0, 4096) for _ in range(random.randint(1, 20))]

    def winner(self):
        return self._winner

    def step(self, move: int):
        self._total_halfmoves += 1
        self._state_prev = self._state
        self._state = np.random.rand(8, 8, 8).astype(np.float32)
        self._done_prev = self._done
        self._done = self._total_halfmoves > 50
        if self._done:
            self._winner_prev = self._winner
            self._winner = random.choice([None, -1, 1])
        return self._done, random.randint(-1, 0)

    def undo(self):
        self._done = self._done_prev
        self._winner = self._winner_prev
        self._total_halfmoves -= 1
        self._state = self._state_prev

    def state_np(self):
        return self._state



class ChessAgent(ABC):
    @abstractmethod
    def get_action(self, game: FakeChessGame) -> int:
        pass


class RandomAgent(ChessAgent):
    def get_action(self, game: FakeChessGame) -> int:
        return random.choice(game.legal_moves())


class ValueNetwork(nn.Module):
    def __init__(self, input_dim=(8, 8, 8)):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x


class ExperienceReplay:
    def __init__(self, capacity=128):
        self.buffer: Deque = collections.deque(maxlen=capacity)

    def push(self, state, next_state, reward):
        self.buffer.append((state, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


class GreedyChessAgent:
    def __init__(self, player=WHITE, lr=1e-3, gamma=0.99, tau=0.001, batch_size=32, device="cpu"):
        self.player = player
        self.device = torch.device(device)

        # Networks
        self.value_net = ValueNetwork().to(self.device)
        self.target_net = ValueNetwork().to(self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())

        # Training parameters
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Experience replay
        self.replay_buffer = ExperienceReplay()

        # Gradient clipping
        self.max_grad_norm = 1.0


    def get_action(self, env: FakeChessGame) -> int:
        assert self.player == WHITE
        assert env.turn == self.player
        legal_moves = list(env.legal_moves())

        if not legal_moves:
            raise ValueError("No legal moves available")

        resulting_states = []
        for move in legal_moves:
            env.step(move)
            resulting_states.append(env.state_np())
            env.undo()

        state_tensors = torch.stack(
            [torch.tensor(state, dtype=torch.float32) for state in resulting_states]
        ).to(self.device)

        with torch.no_grad():
            values = self.value_net(state_tensors).squeeze()

        best_idx = int(torch.argmax(values).item())
        return legal_moves[best_idx]

    def _state_to_tensor(self, state_array: np.ndarray):
        return (
            torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

    def _update_target_network(self):
        for target_param, param in zip(
            self.target_net.parameters(), self.value_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def train_against(
        self,
        opponent: ChessAgent,
        n_episodes: int,
    ):
        pbar = tqdm(range(n_episodes), desc="Training Episodes")
        start = time.time()
        total_steps = 0
        for _ in pbar:
            game = FakeChessGame()
            done = False
            players = {WHITE: self, BLACK: opponent}
            prev_state = game.state_np()
            game_length = 0
            episode_reward = 0.0

            while not done:
                player: ChessAgent = players[game.turn]  # type: ignore
                move = player.get_action(game)
                done, reward = game.step(move)
                total_steps += 1
                state = game.state_np()
                game_length += 1

                if game.turn != self.player:
                    self.add_experience(prev_state, state, reward)
                    episode_reward += reward

                if game.turn == self.player:
                    prev_state = state

                self.train_step()
        end = time.time()
        duration = end - start
        ep_rate = n_episodes / duration
        pos_rate = total_steps / duration
        print(f"Done training in {duration:.2f}s. {ep_rate:.2f} eps/sec, {pos_rate:.2f} moves/sec")

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, next_states, rewards = zip(*batch)

        states_t = torch.stack(
            [torch.tensor(s, dtype=torch.float32) for s in states]
        ).to(self.device)
        next_states_t = torch.stack(
            [torch.tensor(s, dtype=torch.float32) for s in next_states]
        ).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        current_values = self.value_net(states_t).squeeze()

        with torch.no_grad():
            next_values = self.target_net(next_states_t).squeeze()
            target_values = rewards_t + self.gamma * next_values

        loss = F.mse_loss(current_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        self._update_target_network()

        return loss.item()

    def add_experience(self, state, next_state, reward):
        self.replay_buffer.push(state, next_state, reward)


if __name__ == "__main__":
    main()
