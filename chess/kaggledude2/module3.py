""" Training on capture chess from https://www.kaggle.com/code/arjanso/reinforcement-learning-chess-3-q-networks """

import random
import torch
import torch.nn as nn
import torch.optim as optim
from RLC.capture_chess.environment import Board


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.flatten = nn.Flatten(0)
        # 8*8*8 = 8 layers of 8x8 boards, one layer per piece type
        # 64*64 = move piece from X (64 options) to Y (64 options)
        self.stack = nn.Sequential(
            nn.Linear(8*8*8, 64*64, dtype=torch.double)
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.stack(x)
        return x

    def get_action(self, board: Board):
        nn_input = torch.from_numpy(board.layer_board)
        nn_output = self(nn_input)
        action_values = torch.reshape(nn_output, (64, 64))
        legal_mask = torch.from_numpy(board.project_legal_moves())
        action_values = torch.multiply(action_values, legal_mask)
        move_from = torch.argmax(action_values) // 64
        move_to = torch.argmax(action_values) % 64
        moves = [x for x in board.board.generate_legal_moves()
                 if x.from_square == move_from and x.to_square == move_to]
        if len(moves) == 0:
            # If all legal moves have negative action value, explore
            move = board.get_random_action()
            move_from = move.from_square
            move_to = move.to_square
        else:
            # If there are multiple max-moves, pick a random one
            move = random.choice(moves)
        return move


# train loop from torch tute:
# def train_loop(model, loss_fn, optimizer, batch_size, device):
#     # Set the model to training mode - important for batch normalization and
#     # dropout layers Unnecessary in this situation but added for best practices
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X = X.to(device)
#         y = y.to(device)
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * batch_size + len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def play_game(model: LinearModel, board: Board):
    done = False
    i = 0
    while not done:
        action = model.get_action(board)
        done, reward = board.step(action)
        i += 1
        if i > 200:
            raise "dog"
    print("done")


def demo_stuff():
    board = Board()
    print(board.board)
    # each piece type is on a different layer
    print(board.layer_board[0,::-1,:].astype(int))
    print(board.layer_board.shape)
    print(torch.from_numpy(board.layer_board).shape)
    print(nn.Flatten(0)(torch.from_numpy(board.layer_board)).shape)
    action = model.get_action(board)
    print(action)


def learn():
    # RLC/capture_chess/learn.py: Q_learning:
    # play x games. save moves, states, rewards
    # every move, update the model using a random sample of moves etc.
    # every c games, 'fix model' -> save current model weights
        # fixed model is used for making moves
    pass


board = Board()
model = LinearModel()
play_game(model, board)
