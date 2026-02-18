import random
import time

from env.env import ChessGame


def main():
    rando_games_per_sec()


def play_one_game():
    game = ChessGame()
    done = False
    moves = 0
    while not done:
        move = random.choice(list(game.legal_moves()))
        done, reward = game.step(move)
        moves += 1
    return moves


def rando_games_per_sec():
    moves = 0
    games = 0
    start_time = time.perf_counter()
    print_time = time.perf_counter()

    try:
        while True:
            moves += play_one_game()
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
