from lib.agent import RandomAgent
import lib.env as env
from lib.mcts import MctsAgent, random_rollout_reward


def test_mctsrr_vs_random():
    agents = {
        env.WHITE: MctsAgent(env.WHITE, n_sims=2, valfn=lambda e,p: random_rollout_reward(e, p, max_depth=2)),
        env.BLACK: RandomAgent(env.BLACK)
    }
    game = env.ChessGame()
    assert game.turn == env.WHITE
    i = 0
    while not game.is_game_over() and i < 20:
        i += 1
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)
