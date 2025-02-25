""" Training on capture chess from https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks """

import chess
from chess.pgn import Game
import torch
import torch.nn as nn
import torch.optim as optim
import RLC
from RLC.capture_chess.environment import Board


class ConvModel(nn.Module):
    def __init__(self, lr):
        super(ConvModel, self).__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.0)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        inter_layer_1 = self.conv1(x)  # (batch, 1, 8, 8)
        inter_layer_2 = self.conv2(x)  # (batch, 1, 8, 8)
        flat_1 = inter_layer_1.view(x.size(0), 1, -1)  # (batch, 1, 64)
        flat_2 = inter_layer_2.view(x.size(0), 1, -1)  # (batch, 1, 64)
        output_dot_layer = torch.bmm(flat_1, flat_2.transpose(1, 2))  # Batch matrix multiplication
        output_layer = output_dot_layer.view(x.size(0), -1)  # (batch, 4096)
        return output_layer

    def predict(self, state):
        self.eval()
        with torch.no_grad():
            return self.forward(state)

    def get_action(self, board: Board):
        """ My addition, based on RLC/capture_chess/learn.py """
        out = self.predict(board.layer_board)


# def train_on_chess_games(model, num_games=200):
#     for _ in range(num_games):
#         board = Board()
#         while not board.is_game_over():
#             # result = engine.play(board, chess.engine.Limit(time=0.1))
#             board.push(result.move)

#             # Convert board state to tensor
#             board_tensor = torch.rand(1, 8, 8, 8)  # Placeholder for actual board encoding
#             target = torch.rand(1, 4096)  # Placeholder for target output

#             # Forward pass
#             output = model(board_tensor)
#             loss = model.criterion(output, target)

#             # Backpropagation
#             model.optimizer.zero_grad()
#             loss.backward()
#             model.optimizer.step()

# Initialize model and train
# model = ConvModel(lr=0.01)
# train_on_chess_games(model)

###### END CHAT GPT CODE #####################

board = Board()
print(board.board)
# each piece type is on a different layer
print(board.layer_board[0,::-1,:].astype(int))
model = ConvModel(lr=0.01)
action = model.get_action(board)
print(action)
