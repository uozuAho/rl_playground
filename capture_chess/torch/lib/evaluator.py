import time
from RLC.capture_chess.environment import Board  # type: ignore

from lib.nets import ChessNet


def play_game(player: ChessNet, device: str):
    """Note that the opponent is built into the environment. It makes random
    moves."""
    board = Board()
    done = False
    total_reward = 0
    while not done:
        action = player.get_action(board, device)
        done, reward = board.step(action)
        total_reward += reward
    return total_reward


def play_games(player: ChessNet, num_games: int, device: str):
    rewards = []
    for i in range(num_games):
        reward = play_game(player, device)
        rewards.append(reward)
    return sum(rewards) / len(rewards)


def evaluate(description: str, player: ChessNet, num_games: int, device: str):
    print(f'evaluating {description} over {num_games} games...')
    start = time.time()
    avg_reward = play_games(player, num_games, device)
    duration_s = time.time() - start
    print(f"played {num_games} games in {duration_s:0.1f}s ({num_games/duration_s:0.2f} games/s)")
    print(f"avg. reward: {avg_reward}")
    return avg_reward
