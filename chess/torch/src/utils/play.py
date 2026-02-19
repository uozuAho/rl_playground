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
    outcomes: defaultdict = defaultdict(int)
    for _ in range(n_games):
        white_reward, game = play_game(agent_w, agent_b, halfmove_limit=halfmove_limit)
        outcome = game.outcome()
        if outcome:
            outcomes[outcome.termination] += 1
            if outcome.winner:
                key = "white win" if outcome.winner == chess.WHITE else "black win"
                outcomes[key] += 1
        else:
            outcomes["None"] += 1
    return outcomes


def play_games_parallel(
    agent_w: ChessAgent,
    agent_b: ChessAgent,
    n_games: int,
    halfmove_limit: int | None = None,
):
    """returns wins,losses,draws from white's perspective"""
    states = [env.ChessGame(halfmove_limit=halfmove_limit) for _ in range(n_games)]
    dones = [False for _ in range(n_games)]
    winners: list[bool | None | env.Player] = [False for _ in range(n_games)]
    turn = env.WHITE
    while not all(dones):
        active_idx = [i for i in range(len(states)) if not dones[i]]
        active_envs = [states[i] for i in active_idx]
        agent = agent_w if turn == env.WHITE else agent_b
        actions = agent.get_actions(active_envs)
        for i, action in zip(active_idx, actions):
            states[i].do(action)
            if states[i].is_game_over():
                dones[i] = True
                winners[i] = states[i].winner()
        turn = env.other_player(turn)
    assert all(x is not False for x in winners)
    w = _countif(winners, lambda x: x == env.WHITE)
    ll = _countif(winners, lambda x: x == env.BLACK)
    d = _countif(winners, lambda x: x is None)
    return w, ll, d


def _countif(seq, pred):
    return sum(1 if pred(x) else 0 for x in seq)
