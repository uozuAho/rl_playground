"""NNs intended for use with alphazero"""

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from agents.alphazero.move_encoding import Codec, Codec4096
from utils import types

from env import env

# todo: maybe centralise this somewhere with NN 2 move encoding
ACTION_SIZE_4096 = 4096


class AzNet(Protocol):
    def pv(self, state: env.ChessGame) -> types.PV:
        pass

    def pv_batch(self, states: list[env.ChessGame]) -> list[types.PV]:
        pass

    def forward_batch(
        self, states: list[env.ChessGame]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the raw NN tensor outputs (p logits, value)"""
        pass

    def eval(self):
        """Set NN to eval mode"""
        pass

    def train(self):
        """Set NN to train mode"""
        pass

    def get_codec(self) -> Codec:
        pass


class ResNet(AzNet):
    def __init__(self, num_res_blocks, num_hidden, device):
        self.model = _ResNetModule(num_res_blocks, num_hidden, device)
        self.device = device

    def pv(self, state: env.ChessGame) -> types.PV:
        enc_state = torch.from_numpy(self.state2np(state)).to(self.device)
        # unsqueeze to batch of 1
        minput = enc_state.unsqueeze(0).to(self.device)
        plogits, val = self.model(minput)
        return plogits.softmax(dim=1).squeeze().tolist(), val.item()

    def pv_batch(self, states: list[env.ChessGame]) -> list[types.PV]:
        plogits, val = self.forward_batch(states)
        vals = val.squeeze().tolist()
        if type(vals) is float:
            vals = [vals]  # annoying torch quirk. I'm probably not using it properly
        return list(zip(plogits.softmax(dim=1).tolist(), vals))

    def forward_batch(self, states: list[env.ChessGame]):
        n = np.stack([self.state2np(s) for s in states])
        t = torch.from_numpy(n).to(self.device)
        return self.model(t)

    def forward_states_tensor(self, t: torch.Tensor):
        return self.model(t)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def get_codec(self) -> Codec:
        return Codec4096()

    @staticmethod
    def load(path: Path, device):
        d = torch.load(path, weights_only=True)
        n = _ResNetModule(d["num_res_blocks"], d["num_hidden"], device)
        n.load_state_dict(d["state_dict"])
        return n

    @staticmethod
    def state2np(state: env.ChessGame):
        return state.state_np()


# stole this from https://github.com/foersterrobert/AlphaZero
class _ResNetModule(nn.Module):
    def __init__(self, num_res_blocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.num_res_blocks = num_res_blocks
        self.num_hidden = num_hidden

        self.startBlock = nn.Sequential(
            nn.Conv2d(8, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [_ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * env.ROWS * env.COLS, ACTION_SIZE_4096),
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * env.ROWS * env.COLS, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

    def save(self, path: Path):
        torch.save(
            {
                "num_res_blocks": self.num_res_blocks,
                "num_hidden": self.num_hidden,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @staticmethod
    def load(path: Path, device):
        d = torch.load(path, weights_only=True)
        n = ResNet(d["num_res_blocks"], d["num_hidden"], device)
        n.model.load_state_dict(d["state_dict"])
        return n


class _ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
