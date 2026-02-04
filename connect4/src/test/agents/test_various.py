from agents.simple import RandomAgent
from utils.play import play_games_parallel


def test_rando_vs_rando():
    play_games_parallel(RandomAgent(), RandomAgent(), 1)
