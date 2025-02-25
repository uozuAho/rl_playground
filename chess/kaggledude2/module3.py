""" Training on capture chess from https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks """

import chess
from chess.pgn import Game
import torch
import torch.nn as nn
import torch.optim as optim
import RLC
from RLC.capture_chess.environment import Board


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten(0)
        self.stack = nn.Sequential(
            nn.Linear(8*8*8, 4096, dtype=torch.double)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.stack(x)
        return x

    def get_action(self, board: Board):
        input = torch.from_numpy(board.layer_board)
        out = self(input)
        return out


class ConvModel(nn.Module):
    def __init__(self, lr):
        super(ConvModel, self).__init__()
        self.lr = lr
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        inter_layer_1 = self.conv1(x)
        inter_layer_2 = self.conv2(x)
        flat_1 = inter_layer_1.view(x.size(0), 1, -1)
        flat_2 = inter_layer_2.view(x.size(0), 1, -1)
        output_dot_layer = torch.bmm(flat_1, flat_2.transpose(1, 2))
        output_layer = output_dot_layer.view(x.size(0), -1)
        return output_layer

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            return self.forward(state)

    def get_action(self, board: Board):
        """ My addition, based on RLC/capture_chess/learn.py """
        out = self.predict(board.layer_board)


board = Board()
# print(board.board)
# each piece type is on a different layer
# print(board.layer_board[0,::-1,:].astype(int))
# print(board.layer_board.shape)
# print(torch.from_numpy(board.layer_board).shape)
# print(nn.Flatten(0)(torch.from_numpy(board.layer_board)).shape)
model = LinearModel()
action = model.get_action(board)
print(action)
