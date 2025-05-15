import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib.env import CaptureChess
from lib.nets import ChessNet


class ConvPolicyNet(ChessNet):
    def __init__(self):
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def forward(self, x, legal_moves):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x1_flat = x1.view(-1, 1, 64)
        x2_flat = x2.view(-1, 64, 1)

        dot = torch.bmm(x1_flat, x2_flat).view(-1, 4096)
        softmaxed = F.softmax(dot, dim=1)
        masked = softmaxed * legal_moves

        return masked


class PolicyGradientTrainer:
    def __init__(self, lr=0.01, gamma=0.99):
        self.model = ConvPolicyNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.long_term_mean = []
        self.verbose = 1
        self.weight_memory = []

    def epsilon(eps_start, eps_end, n_total_ep, ep):
        return eps_end + (eps_start - eps_end) * math.exp(-6.0 * ep / n_total_ep)

    def train(self, n_episodes=1):
        for _ in range(n_episodes):
            states, actions, rewards, action_spaces = self.play_game()
            self.policy_gradient_update(states, actions, rewards, action_spaces)

    def play_game(self):
        game = CaptureChess(action_limit=25)
        done = False

        # 8x8x8 boards
        states: list[np.ndarray] = []  # type: ignore
        # [move (from, to)]
        actions: list[tuple[int, int]] = []  # type: ignore
        rewards: list[float] = []  # type: ignore
        # legal moves
        legal_moves: list[np.ndarray] = []  # type: ignore

        while not done:
            state = game.layer_board
            legal_move_mask = game.project_legal_moves()
            with torch.no_grad():
                action_probs = self.model(
                    [
                        np.expand_dims(state, axis=0),
                        np.zeros((1, 1)),
                        legal_move_mask.reshape(1, 4096),
                    ]
                )
            # self.action_value_mem.append(action_probs)
            action_probs = action_probs / action_probs.sum()
            move: int = np.random.choice(range(4096), p=np.squeeze(action_probs))  # type: ignore
            move_from = move // 64
            move_to = move % 64
            moves = [
                x
                for x in game.board.generate_legal_moves()
                if x.from_square == move_from and x.to_square == move_to
            ]
            assert len(moves) > 0
            if len(moves) > 1:
                # If there are multiple max-moves, pick a random one.
                move = np.random.choice(moves)
            else:
                move = moves[0]

            done, reward = game.step(move)
            new_state = game.layer_board
            if done:
                new_state = new_state * 0

            states.append(state)
            actions.append((move_from, move_to))
            rewards.append(reward)
            legal_moves.append(legal_move_mask.reshape(1, 4096))

        return states, actions, rewards, legal_moves

    def policy_gradient_update(
        self,
        states: list[np.ndarray],
        actions: list[tuple[int, int]],
        rewards: list[float],
        legal_moves: list[np.ndarray],
    ):
        """
        Update the network with data from a full episode.

        Params:
            - states: list of board states (8x8x8 array)
            - actions: list of move (from, to)
            - rewards: list of rewards
            - legal_moves: list of legal move masks (1x4096 array)
        """
        self.model.train()
        n_steps = len(states)
        returns = []
        targets = torch.zeros((n_steps, 4096))

        for t in range(n_steps):
            move_from, move_to = actions[t]
            idx = move_from * 64 + move_to
            targets[t, idx] = 1.0
            r = sum([r * (self.gamma**i) for i, r in enumerate(rewards[t:])])
            returns.append(r)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        mean_return = returns_t.mean().item()
        self.long_term_mean.append(mean_return)
        baseline = sum(self.long_term_mean) / len(self.long_term_mean)
        train_returns = returns - baseline

        states_tensor = torch.tensor(states, dtype=torch.float32)

        # todo: confirm this is right. Add types
        # convert from TensorFlow (and Keras) : (batch_size, height, width, channels)
        # to PyTorch                            (batch_size, channels, height, width)
        states_tensor = states_tensor.permute(0, 3, 1, 2)

        action_spaces_tensor = torch.tensor(
            np.concatenate(legal_moves, axis=0), dtype=torch.float32
        )

        self.optimizer.zero_grad()
        probs = self.model(states_tensor, action_spaces_tensor)
        log_probs = torch.log(torch.sum(probs * targets, dim=1) + 1e-10)
        loss = -torch.mean(train_returns * log_probs)
        loss.backward()
        self.optimizer.step()
