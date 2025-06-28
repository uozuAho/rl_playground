import random
import collections
from typing import Deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.agent import ChessAgent
from lib.env import BLACK, WHITE, ChessGame, Player


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

    def push(self, state, next_state, reward):
        self.buffer.append((state, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


class GreedyChessAgent(ChessAgent):
    def __init__(self, player: Player, lr=1e-3, gamma=0.99, tau=0.001, batch_size=32):
        assert player == WHITE
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

        # Training metrics
        self.episode_wins: list[int] = []
        self.episode_game_lengths: list[int] = []
        self.episode_losses: list[float] = []
        self.episode_rewards: list[float] = []
        self.episode_count = 0

    def get_action(self, env: ChessGame) -> chess.Move:
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
        capture_reward_factor=0.0,
        halfmove_limit: int | None = None,
        plot=False,
        print_every=100,
    ):
        pbar = tqdm(range(n_episodes), desc="Training Episodes")
        for episode in pbar:
            game = ChessGame(
                capture_reward_factor=capture_reward_factor,
                halfmove_limit=halfmove_limit,
            )
            done = False
            players: dict[Player, ChessAgent] = {WHITE: self, BLACK: opponent}
            prev_state = game.state_np()
            game_length = 0
            episode_losses = []
            episode_reward = 0.0

            while not done:
                move = players[game.turn].get_action(game)
                done, reward = game.step(move)
                state = game.state_np()
                game_length += 1

                if game.turn != self.player:
                    self.add_experience(prev_state, state, reward)
                    episode_reward += reward

                if game.turn == self.player:
                    prev_state = state

                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

            # Track metrics for this episode
            self.episode_count += 1
            winner = game.winner()
            self.episode_wins.append(1 if winner == WHITE else 0)
            self.episode_game_lengths.append(game_length)
            avg_loss = (
                sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
            )
            self.episode_losses.append(avg_loss)
            self.episode_rewards.append(episode_reward)

            # Update progress bar with current metrics
            if (episode + 1) % 10 == 0:
                recent_wins = self.episode_wins[-min(100, len(self.episode_wins)):]
                win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
                pbar.set_postfix({
                    'Win Rate': f'{win_rate:.3f}',
                    'Avg Loss': f'{avg_loss:.4f}',
                    'Game Len': game_length,
                    'Reward': f'{episode_reward:.2f}'
                })

            # Print detailed metrics periodically
            if (episode + 1) % print_every == 0:
                stats = self.get_training_stats()
                print(f"\nEpisode {episode + 1}/{n_episodes}")
                print(f"  Win Rate (recent): {stats['recent_win_rate']:.3f}")
                print(f"  Win Rate (overall): {stats['overall_win_rate']:.3f}")
                print(f"  Avg Game Length: {stats['recent_avg_game_length']:.1f}")
                print(f"  Avg Loss: {stats['recent_avg_loss']:.4f}")
                print(f"  Avg Reward: {stats['recent_avg_reward']:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")

        if plot:
            self.plot_training_metrics()

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

    def plot_training_metrics(self, window_size=100):
        if not self.episode_wins:
            print("No training data to plot")
            return

        episodes = list(range(1, len(self.episode_wins) + 1))

        # Calculate rolling averages
        def rolling_average(data, window):
            if len(data) < window:
                return data
            return [
                sum(data[i : i + window]) / window
                for i in range(len(data) - window + 1)
            ]

        win_rate_rolling = rolling_average(self.episode_wins, window_size)
        game_length_rolling = rolling_average(self.episode_game_lengths, window_size)
        loss_rolling = rolling_average(self.episode_losses, window_size)
        reward_rolling = rolling_average(self.episode_rewards, window_size)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

        ax1.plot(
            episodes[-len(win_rate_rolling) :], win_rate_rolling, "b-", linewidth=2
        )
        ax1.set_ylabel("Win Rate")
        ax1.set_title(f"Training Performance (Rolling Average, Window={window_size})")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        ax2.plot(
            episodes[-len(game_length_rolling) :],
            game_length_rolling,
            "g-",
            linewidth=2,
        )
        ax2.set_ylabel("Average Game Length")
        ax2.grid(True, alpha=0.3)

        ax3.plot(episodes[-len(loss_rolling) :], loss_rolling, "r-", linewidth=2)
        ax3.set_ylabel("Evaluation Loss")
        ax3.grid(True, alpha=0.3)

        ax4.plot(episodes[-len(reward_rolling) :], reward_rolling, "m-", linewidth=2)
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Average Reward")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_training_stats(self):
        if not self.episode_wins:
            return {}

        recent_episodes = min(100, len(self.episode_wins))
        recent_wins = self.episode_wins[-recent_episodes:]
        recent_lengths = self.episode_game_lengths[-recent_episodes:]
        recent_losses = self.episode_losses[-recent_episodes:]
        recent_rewards = self.episode_rewards[-recent_episodes:]

        return {
            "total_episodes": len(self.episode_wins),
            "recent_win_rate": sum(recent_wins) / len(recent_wins),
            "recent_avg_game_length": sum(recent_lengths) / len(recent_lengths),
            "recent_avg_loss": sum(recent_losses) / len(recent_losses),
            "recent_avg_reward": sum(recent_rewards) / len(recent_rewards),
            "overall_win_rate": sum(self.episode_wins) / len(self.episode_wins),
        }
