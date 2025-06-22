import lib.env as env
from lib.env import ChessGame, WHITE, BLACK
from lib.agent import ChessAgent, RandomAgent
from lib.greedy_agent import GreedyChessAgent


def test_greedy_agent():
    agent = GreedyChessAgent(WHITE, batch_size=4)

    players: dict[env.Player, ChessAgent] = {  # type: ignore
        WHITE: agent,
        BLACK: RandomAgent(BLACK)
    }

    game = ChessGame()
    assert game.turn == WHITE

    for _ in range(agent.batch_size * 2):
        move = players[game.turn].get_action(game)
        game_over, reward = game.step(move)
        assert not game_over
        if game.turn == WHITE:
            agent.add_experience(game.state_np(), reward)

    agent.train_step()
