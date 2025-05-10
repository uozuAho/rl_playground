from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ChessNet(ABC):
    pass


class LinearFC(nn.Module, ChessNet):
    """Fully connected linear/sequential NN"""

    def __init__(self):
        super(LinearFC, self).__init__()
        self.flatten = nn.Flatten()
        # 8*8*8 = 8 layers of 8x8 boards, one layer per piece type
        # 64*64 = move piece from X (64 options) to Y (64 options)
        # eg output indexes:
        # 0: move 0 to 0
        # 1: move 0 to 1
        # ...
        # 4094: move 63 to 62
        # 4095: move 63 to 63
        self.stack = nn.Sequential(nn.Linear(8 * 8 * 8, 64 * 64, dtype=torch.float64))

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.stack(x)
        return x
