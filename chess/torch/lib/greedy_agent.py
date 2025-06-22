import random
import collections
from typing import Deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess

from lib.agent import ChessAgent
from lib.env import ChessGame, Player


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
    def __init__(self, capacity=10000):
        self.buffer: Deque = collections.deque(maxlen=capacity)

    def push(self, state, reward):
        self.buffer.append((state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


class GreedyChessAgent(ChessAgent):
    def __init__(self, player: Player, lr=1e-3, gamma=0.99, tau=0.001, batch_size=32):
        self.player = player
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def get_action(self, env: ChessGame) -> chess.Move:
        assert env.turn == self.player
        legal_moves = list(env.legal_moves())

        if not legal_moves:
            raise ValueError("No legal moves available")

        best_move = None
        best_value = float('-inf') if self.player == 1 else float('inf')

        for move in legal_moves:
            # Make a copy and simulate the move
            env_copy = env.copy()
            env_copy.step(move)

            # Get the value of the resulting position
            state_tensor = self._state_to_tensor(env_copy.state_np())
            with torch.no_grad():
                value = self.value_net(state_tensor).item()

            # For WHITE (player=1), we want to maximize value
            # For BLACK (player=-1), we want to minimize value
            if self.player == 1:  # WHITE
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # BLACK
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_move if best_move else legal_moves[0]

    def _state_to_tensor(self, state_array):
        """Convert numpy state array to torch tensor"""
        return torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)

    def update_target_network(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, rewards = zip(*batch)

        # Convert to tensors
        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Get current values
        current_values = self.value_net(state_batch).squeeze()

        # Compute loss (MSE between predicted value and actual reward)
        loss = F.mse_loss(current_values, reward_batch)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Update target network
        self.update_target_network()

        return loss.item()

    def add_experience(self, state, reward):
        """Add experience to replay buffer"""
        self.replay_buffer.push(state, reward)

    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'value_net_state_dict': self.value_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
