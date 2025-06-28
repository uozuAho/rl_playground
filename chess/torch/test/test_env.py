import random

import chess
import numpy as np
from lib.agents.agent import RandomAgent
import lib.env as env


def test_random_game():
    game = env.ChessGame(halfmove_limit=20)
    num_moves = 0
    while not game.is_game_over():
        move = random.choice(list(game.legal_moves()))
        game_over, _ = game.step(move)
        num_moves += 1
        assert game_over == game.is_game_over()
    assert num_moves <= 20


def test_random_agent():
    agents = {env.WHITE: RandomAgent(env.WHITE), env.BLACK: RandomAgent(env.BLACK)}
    game = env.ChessGame(halfmove_limit=20)
    assert game.turn == env.WHITE
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)


def test_initial_state():
    game = env.ChessGame()
    state = game.state_np()
    assert state.shape == (8, 8, 8)
    assert np.all(state[0, 1, :] == 1)  # white pawns on row 1
    assert np.all(state[0, 6, :] == -1)  # black pawns on row 7


def test_white_captures_black_positive_reward():
    game = env.ChessGame(
        fen="8/8/8/8/8/6p1/7P/8 w KQkq - 0 1", capture_reward_factor=0.1
    )
    _, reward = game.step(chess.Move(chess.H2, chess.G3))
    assert reward > 0


def test_black_captures_white_positive_reward():
    game = env.ChessGame(
        fen="8/8/8/8/8/6p1/7P/8 b KQkq - 0 1", capture_reward_factor=0.1
    )
    _, reward = game.step(chess.Move(chess.G3, chess.H2))
    assert reward < 0
