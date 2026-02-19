from env import env
from env.env import ChessGame, WHITE, BLACK
from agents.agent import ChessAgent
from agents.random import RandomAgent
from agents.greedy_agent import GreedyChessAgent


def test_greedy_agent():
    agent = GreedyChessAgent(WHITE, batch_size=4)

    players: dict[env.Player, ChessAgent] = {
        WHITE: agent,
        BLACK: RandomAgent(BLACK),
    }

    game = ChessGame()
    assert game.turn == WHITE

    prev_state = game.state_np()

    for _ in range(agent.batch_size * 2):
        move = players[game.turn].get_action(game)
        game.do(move)
        state = game.state_np()
        assert not game.is_game_over()
        if game.turn == BLACK:
            agent.add_experience(prev_state, state, 1.0)
        prev_state = state

    agent.train_step()
