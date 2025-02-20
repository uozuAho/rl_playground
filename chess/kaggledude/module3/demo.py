""" Capture chess. Aim is to capture pieces. """

import chess
from chess.pgn import Game
import RLC
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Q_learning
from RLC.capture_chess.agent import Agent


board = Board()

# render the board
# print(board.board)

# state = 8x8x8 array. layer 0 = pawns, 1 = rooks etc
# white pieces are 1, black are -1
# print(board.layer_board[0,::-1,:].astype(int))

agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
R.agent.fix_model()
# R.agent.model.summary()

pgn = R.learn(iters=200)
