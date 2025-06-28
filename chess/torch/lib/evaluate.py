from collections import Counter, defaultdict
import time

import chess
from lib import env
from lib.agents.agent import ChessAgent


def play_game(
    agent_w: ChessAgent, agent_b: ChessAgent, halfmove_limit: int | None = None
):
    agents = {env.WHITE: agent_w, env.BLACK: agent_b}
    game = env.ChessGame(halfmove_limit=halfmove_limit)
    done = False
    total_white_reward = 0.0
    while not done:
        agent = agents[game.turn]
        move = agent.get_action(game)
        done, white_reward = game.step(move)
        total_white_reward += white_reward
    return total_white_reward, game


def evaluate(
    agent_w: ChessAgent,
    agent_b: ChessAgent,
    n_games=10,
    halfmove_limit: int | None = None,
):
    start = time.time()
    print(f"evaulating over {n_games} games...")
    white_rewards = []
    game_lens = []
    outcomes: defaultdict = defaultdict(int)
    for _ in range(n_games):
        white_reward, game = play_game(agent_w, agent_b)
        outcome = game.outcome
        game_len = len(game._board.move_stack)
        game_lens.append(game_len)
        white_rewards.append(white_reward)
        if outcome:
            outcomes[outcome.termination] += 1
            if outcome.winner:
                key = 'white win' if outcome.winner == chess.WHITE else 'black win'
                outcomes[key] += 1
        else:
            outcomes['None'] += 1
    total_time = time.time() - start
    games_sec = n_games / total_time
    print(f"Played {n_games} games in {total_time:0.1f}s ({games_sec:0.2f} games/s)")
    print(f"Avg white reward {sum(white_rewards) / len(white_rewards)}")
    print("Outcomes:")
    for k, v in outcomes.items():
        print(f'{k}: {v}')
