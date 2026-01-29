from agents.perfect import PerfectAgent
from agents.random import RandomAgent
from utils.ranker import AgentRanker


def test_ranker():
    agents = [
        ("Random 1", RandomAgent()),
        ("Random 2", RandomAgent()),
        ("Perfect", PerfectAgent()),
    ]

    ranker = AgentRanker(agents)
    stats = ranker.full_round_robin(games_per_matchup=3)

    best = max(stats, key=lambda s: s.score())
    assert best.name == "Perfect"
