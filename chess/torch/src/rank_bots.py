"""Intended to replace bot showdown.
Train bots elsewhere - this should just be a tournament.
"""

import agents.mctsnew as mcts
from agents.random import RandomAgent
from env.env import WHITE
# disable unused import check so u can temporarily comment out bots
# ruff: noqa: F401

from utils import ranker


def main():
    agents = [
        ("Random1", RandomAgent(WHITE)),
        ("MctsU10", mcts.make_uniform_agent(10)),
    ]
    stats = ranker.full_round_robin(agents, games_per_matchup=5)
    ranker.print_rankings(stats)


if __name__ == "__main__":
    main()
