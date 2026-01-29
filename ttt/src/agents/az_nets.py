"""NNs intended for use with alphazero"""

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


# stole this from https://github.com/foersterrobert/AlphaZero
class ResNet(nn.Module):
    def __init__(self, num_res_blocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.num_res_blocks = num_res_blocks
        self.num_hidden = num_hidden

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
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
            nn.Linear(32 * 3 * 3, 9),
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 3, 1),
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
        n.load_state_dict(d["state_dict"])
        return n


class _ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # TODO: are num channels here correct? check ospeil
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
