import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import torchinfo

from lib.env import CaptureChess


class ConvPolicyNet(nn.Module):
    def __init__(self):
        super(ConvPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64
        )

    def print_summary(self, device):
        input_data = {
            "x": torch.ones((1, 8, 8, 8), dtype=torch.float64),
            "legal_moves": torch.ones((1, 4096), dtype=torch.float64),
        }
        torchinfo.summary(
            self, input_data=input_data, dtypes=[torch.float64], device=device
        )

    def forward(self, x: torch.Tensor, legal_moves: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1_flat = x1.view(x1.size(0), 64, 1)
        x2_flat = x2.view(x2.size(0), 1, 64)
        cross = torch.bmm(x1_flat, x2_flat)
        cross = cross.view(cross.size(0), -1)
        softmaxed = F.softmax(cross, dim=1)
        masked = softmaxed * legal_moves
        return masked

    # todo: dedupe this and train. Train selects move probabilistically
    def get_action(self, game: CaptureChess, device: str):
        nn_input = torch.from_numpy(game.layer_board).unsqueeze(0).to(device)
        legal_moves = (
            torch.from_numpy(game.project_legal_moves()).reshape((1, 4096)).to(device)
        )
        with torch.no_grad():
            nn_output: torch.Tensor = self(nn_input, legal_moves)
        move_idx = nn_output.argmax().item()
        move_from = move_idx // 64
        move_to = move_idx % 64
        moves = [
            x
            for x in game.board.generate_legal_moves()
            if x.from_square == move_from and x.to_square == move_to
        ]
        assert len(moves) > 0
        return moves[0]


class PolicyGradientTrainer:
    def __init__(self, lr=0.01, gamma=0.99, device="cpu"):
        self.model = ConvPolicyNet().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.long_term_mean = []
        self.verbose = 1
        self.weight_memory = []

    def epsilon(eps_start, eps_end, n_total_ep, ep):
        return eps_end + (eps_start - eps_end) * math.exp(-6.0 * ep / n_total_ep)

    @staticmethod
    def train_known_sample(
        states: list[np.ndarray],
        actions: list[tuple[int, int]],
        rewards: list[float],
        legal_moves: list[np.ndarray],
        n_steps: int,
        device: str
    ):
        """ For debugging """
        trainer = PolicyGradientTrainer(device=device)
        for _ in range(n_steps):
            loss = trainer.policy_gradient_update(states, actions, rewards, legal_moves, device)
            print(loss)
        return trainer.model

    @staticmethod
    def train_new_net(n_episodes=1, device="cpu"):
        trainer = PolicyGradientTrainer(device=device)
        ep_avg_losses = []
        ep_total_rewards = []
        for _ in range(n_episodes):
            states, actions, rewards, action_spaces = trainer.play_game(device)
            loss = trainer.policy_gradient_update(
                states, actions, rewards, action_spaces, device
            )
            ep_avg_losses.append(loss)
            ep_total_rewards.append(sum(rewards))
        return trainer.model, ep_avg_losses, ep_total_rewards

    def play_game(self, device: str):
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
                    # np.expand_dims(state, axis=0),
                    torch.from_numpy(state).unsqueeze(0).to(device),
                    torch.from_numpy(legal_move_mask.reshape(1, 4096)).to(device),
                )
                dist = Categorical(action_probs)
                move_idx = dist.sample().item()
            # self.action_value_mem.append(action_probs)
            # action_probs = action_probs / action_probs.sum()
            # move_idx: int = np.random.choice(
            #     range(4096), p=np.squeeze(action_probs.cpu())
            # )  # type: ignore
            move_from = move_idx // 64
            move_to = move_idx % 64
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
            legal_moves.append(legal_move_mask.reshape(4096,))

        return states, actions, rewards, legal_moves

    def policy_gradient_update(
        self,
        states: list[np.ndarray],
        actions: list[tuple[int, int]],
        rewards: list[float],
        legal_moves: list[np.ndarray],
        device: str,
    ):
        """
        Update the network with data from a full episode.

        Params:
            - states: list of board states (8x8x8 array)
            - actions: list of move (from, to)
            - rewards: list of rewards
            - legal_moves: list of legal move masks (4096 element array)
        """
        self.model.train()
        n_steps = len(states)
        returns = []
        targets = torch.zeros((n_steps, 4096)).to(device)

        for t in range(n_steps):
            move_from, move_to = actions[t]
            idx = move_from * 64 + move_to
            targets[t, idx] = 1.0
            r = sum([r * (self.gamma**i) for i, r in enumerate(rewards[t:])])
            returns.append(r)

        returns_t = torch.tensor(returns, dtype=torch.float64).to(device)
        mean_return = returns_t.mean().item()
        self.long_term_mean.append(mean_return)
        baseline = sum(self.long_term_mean) / len(self.long_term_mean)
        train_returns = returns_t - baseline

        # states_t = torch.tensor(states, dtype=torch.float64)
        # stack? concat? I never remember
        states_t = torch.stack([torch.from_numpy(s) for s in states]).to(device)

        # todo: confirm this is right. Add types
        # convert from TensorFlow (and Keras) : (batch_size, height, width, channels)
        # to PyTorch                            (batch_size, channels, height, width)
        # states_t = states_t.permute(0, 3, 1, 2)

        # action_spaces_tensor = torch.tensor(
        #     np.concatenate(legal_moves, axis=0), dtype=torch.float64
        # ).to(device)

        legal_moves_t = torch.stack([torch.from_numpy(x) for x in legal_moves]).to(device)

        self.optimizer.zero_grad()
        probs = self.model(states_t, legal_moves_t)
        log_probs = torch.log(torch.sum(probs * targets, dim=1) + 1e-10)
        loss = -torch.mean(train_returns * log_probs)
        loss.backward()
        self.optimizer.step()

        return loss.item()
