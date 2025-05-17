from lib.agent import RandomAgent
import lib.env as env
from lib.mcts import MctsAgent


def test_mctsrr_vs_random():
    agents = {
        env.WHITE: MctsAgent(env.WHITE, n_sims=10),
        env.BLACK: RandomAgent(env.BLACK)
    }
    game = env.ChessGame()
    assert game.turn == env.WHITE
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)
