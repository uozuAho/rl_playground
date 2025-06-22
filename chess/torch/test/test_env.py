import random

import numpy as np
from lib.agent import RandomAgent
import lib.env as env


def test_random_game():
    game = env.ChessGame()
    while not game.is_game_over():
        move = random.choice(list(game.legal_moves()))
        game.step(move)


def test_random_agent():
    agents = {
        env.WHITE: RandomAgent(env.WHITE),
        env.BLACK: RandomAgent(env.BLACK)
    }
    game = env.ChessGame()
    assert game.turn == env.WHITE
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)


def test_initial_state():
    game = env.ChessGame()
    state = game.state_np()
    assert state.shape == (8,8,8)
    assert np.all(state[0, 1, :] == 1)  # white pawns on row 1
    assert np.all(state[0, 6, :] == -1)  # black pawns on row 7
