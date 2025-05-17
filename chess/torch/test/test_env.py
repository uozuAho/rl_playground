import random
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
