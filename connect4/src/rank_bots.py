"""Intended to replace bot showdown.
Train bots elsewhere - this should just be a tournament.
"""
from agents.simple import RandomAgent
from utils import ranker


def main():
    agents = [
        ("Random1", RandomAgent()),
        ("Random2", RandomAgent()),
    ]
    stats = ranker.full_round_robin(agents, games_per_matchup=20)
    ranker.print_rankings(stats)


if __name__ == "__main__":
    main()
