from env import env
from agents.agent import RandomAgent
from agents.andoma.andoma_agent import AndomaAgent


def test_random_agent():
    agents = {env.WHITE: AndomaAgent(env.WHITE), env.BLACK: RandomAgent(env.BLACK)}
    game = env.ChessGame(halfmove_limit=20)
    assert game.turn == env.WHITE
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)
