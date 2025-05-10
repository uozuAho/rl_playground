from RLC.capture_chess.environment import Board  # type: ignore

from lib.nets import ChessNet


def play_game(player: ChessNet, device: str):
    """ Note that the opponent is built into the environment. It makes random
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
    return sum(rewards)/len(rewards)

