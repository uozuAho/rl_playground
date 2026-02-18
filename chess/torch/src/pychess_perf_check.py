import time

# disable unused import check so u can temporarily comment out bots
# ruff: noqa: F401

from agents.agent import ChessAgent
from agents.andoma.andoma_agent import AndomaAgent
from agents.mcts import MctsAgent, random_rollout_reward
from agents.random import RandomAgent
from env.env import ChessGame, WHITE, BLACK


def main():
    print_games_per_sec(RandomAgent(WHITE), RandomAgent(BLACK))

    # print_games_per_sec(
    #     MctsAgent(
    #         WHITE,
    #         n_sims=10,
    #         valfn=lambda e, p: random_rollout_reward(e, p, max_depth=5),
    #     ),
    #     RandomAgent(BLACK),
    # )

    # print_games_per_sec(AndomaAgent(WHITE, search_depth=1), RandomAgent(BLACK))


def play_one_game(agent_w: ChessAgent, agent_b: ChessAgent):
    game = ChessGame()
    moves = 0
    while not game.is_game_over():
        agent = agent_w if game.turn == WHITE else agent_b
        move = agent.get_action(game)
        game.do(move)
        moves += 1
    return moves


def print_games_per_sec(agent_w: ChessAgent, agent_b: ChessAgent):
    moves = 0
    games = 0
    start_time = time.perf_counter()
    print_time = time.perf_counter()

    try:
        while True:
            moves += play_one_game(agent_w, agent_b)
            games += 1
            if time.perf_counter() - print_time >= 1.0:
                print_time = time.perf_counter()
                elapsed = time.perf_counter() - start_time
                print(
                    f"{moves / elapsed:.2f} moves/sec   {games / elapsed:.2f} games/sec"
                )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
