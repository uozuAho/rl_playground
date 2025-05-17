import random
import lib.env as env


def test_random_game():
    game = env.ChessGame()
    while not game.is_game_over():
        move = random.choice(list(game.legal_moves()))
        game.step(move)
