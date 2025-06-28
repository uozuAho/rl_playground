from lib.agents.agent import RandomAgent
import lib.env as env
from lib.agents.mcts import MctsAgent, random_rollout_reward


def test_mctsrr_vs_random():
    agents = {
        env.WHITE: MctsAgent(
            env.WHITE,
            n_sims=2,
            valfn=lambda e, p: random_rollout_reward(e, p, max_depth=2),
        ),
        env.BLACK: RandomAgent(env.BLACK),
    }
    game = env.ChessGame(halfmove_limit=20)
    assert game.turn == env.WHITE
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.step(move)
