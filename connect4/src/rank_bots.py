"""Intended to replace bot showdown.
Train bots elsewhere - this should just be a tournament.
"""

import agents.mcts_agent as mcts
# disable unused import check so u can temporarily comment out bots
# ruff: noqa: F401

from agents.simple import RandomAgent, FirstLegalActionAgent
from utils import ranker


def main():
    agents = [
        ("Random1", RandomAgent()),
        # ("Random2", RandomAgent()),
        ("FirstLegal", FirstLegalActionAgent()),
        ("MctsU10", mcts.make_uniform_agent(10)),
        ("MctsU20", mcts.make_uniform_agent(20)),
    ]
    stats = ranker.full_round_robin(agents, games_per_matchup=100)
    ranker.print_rankings(stats)


if __name__ == "__main__":
    main()
