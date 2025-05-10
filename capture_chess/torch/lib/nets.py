from abc import ABC
import chess
import torch
import torch.nn as nn
import torchinfo
from RLC.capture_chess.environment import Board  # type: ignore


class ChessNet(nn.Module, ABC):
    def print_summary(self):
        torchinfo.summary(self, input_size=(8, 8, 8), dtypes=[torch.float64])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def get_action(self, board: Board, device: str) -> chess.Move:
        """Assumes a net with a 1x4096 (64x64) output, which represents a
        move from (64) -> to (64)
        """
        nn_input = torch.from_numpy(board.layer_board).unsqueeze(0).to(device)
        with torch.no_grad():
            nn_output = self(nn_input)
        action_values = torch.reshape(nn_output, (64, 64))
        legal_mask = torch.from_numpy(board.project_legal_moves()).to(device)
        action_values = torch.multiply(action_values, legal_mask)
        move_from = torch.argmax(action_values) // 64
        move_to = torch.argmax(action_values) % 64
        moves = [
            x
            for x in board.board.generate_legal_moves()
            if x.from_square == move_from and x.to_square == move_to
        ]
        if len(moves) == 0:
            print(action_values)
            print(move_from)
            print(move_to)
            print(board.board)
            raise Exception("no valid moves")
        else:
            return moves[0]


class LinearFCQNet(ChessNet):
    """Fully connected linear/sequential NN"""

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

    @staticmethod
    def load(path: str, device: str):
        net = ConvQNet().to(device)
        net.load_state_dict(torch.load(path, weights_only=True))
        return net
