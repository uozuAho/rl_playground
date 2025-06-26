import lib.env as env
from lib.env import ChessGame, WHITE, BLACK
from lib.agent import ChessAgent, RandomAgent
from lib.greedy_agent import GreedyChessAgent


def test_greedy_agent():
    agent = GreedyChessAgent(WHITE, batch_size=4)

    players: dict[env.Player, ChessAgent] = {  # type: ignore
        WHITE: agent,
        BLACK: RandomAgent(BLACK),
    }

    game = ChessGame()
    assert game.turn == WHITE

    prev_state = game.state_np()

    for _ in range(agent.batch_size * 2):
        move = players[game.turn].get_action(game)
        game_over, reward = game.step(move)
        state = game.state_np()
        assert not game_over
        if game.turn == BLACK:
            agent.add_experience(prev_state, state, reward)
        prev_state = state

    agent.train_step()
