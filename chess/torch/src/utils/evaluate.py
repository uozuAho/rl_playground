from collections import defaultdict

import chess
from env import env
from agents.agent import ChessAgent


def play_game(
    agent_w: ChessAgent, agent_b: ChessAgent, halfmove_limit: int | None = None
):
    agents = {env.WHITE: agent_w, env.BLACK: agent_b}
    game = env.ChessGame(halfmove_limit=halfmove_limit)
    while not game.is_game_over():
        agent = agents[game.turn]
        move = agent.get_action(game)
        game.do(move)
    return game.outcome()


def play_games(
    agent_w: ChessAgent,
    agent_b: ChessAgent,
    n_games=10,
    halfmove_limit: int | None = None,
):
    game_lens = []
    outcomes: defaultdict = defaultdict(int)
    for _ in range(n_games):
        white_reward, game = play_game(agent_w, agent_b, halfmove_limit=halfmove_limit)
        outcome = game.outcome()
        game_len = len(game._board.move_stack)
        game_lens.append(game_len)
        if outcome:
            outcomes[outcome.termination] += 1
            if outcome.winner:
                key = "white win" if outcome.winner == chess.WHITE else "black win"
                outcomes[key] += 1
        else:
            outcomes["None"] += 1
    return outcomes
