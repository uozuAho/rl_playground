from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class ChessNet(nn.Module, ABC):
    def print_summary(self):
        print("TODO")


class LinearFCQNet(ChessNet):
    """Fully connected linear/sequential NN. output = 64x64"""

    def __init__(self):
        super(LinearFCQNet, self).__init__()
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


class ConvQNet(ChessNet):
    """Dunno if this is correct, I just got chat gpt to convert from TF to torch:
    https://github.com/arjangroen/RLC/blob/e54eb7380875f64fd06106c59aa376b426d9e5ca/RLC/capture_chess/agent.py#L73

    This trains much quicker than the linear FC net, as the conv layers reduce
    dimensionality by 8 (I think).
    """

    def __init__(self):
        super(ConvQNet, self).__init__()

        # 1x1 conv layers used to blend input layers
        self.conv1 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1, dtype=torch.float64
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1_flat = x1.view(x1.size(0), 64, 1)
        x2_flat = x2.view(x2.size(0), 1, 64)
        output = torch.bmm(x1_flat, x2_flat)
        output = output.view(output.size(0), -1)
        return output
