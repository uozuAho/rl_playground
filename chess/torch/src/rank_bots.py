"""Intended to replace bot showdown.
Train bots elsewhere - this should just be a tournament.
"""

import agents.mctsnew as mcts
from agents.andoma.andoma_agent import AndomaAgent
from agents.random import RandomAgent
# disable unused import check so u can temporarily comment out bots
# ruff: noqa: F401

from utils import ranker


def main():
    agents = [
        ("Random", RandomAgent()),
        # ("MctsU", mcts.make_uniform_agent(10)),
        ("Andoma", AndomaAgent(search_depth=1))
    ]
    stats = ranker.full_round_robin(agents, games_per_matchup=5)
    ranker.print_rankings(stats)


if __name__ == "__main__":
    main()
