from __future__ import annotations
from dataclasses import dataclass

from agents.agent import Agent
from utils.play import play_games_parallel


@dataclass
class AgentStats:
    agent: Agent
    name: str
    x_wins: int = 0
    x_losses: int = 0
    x_draws: int = 0
    o_wins: int = 0
    o_losses: int = 0
    o_draws: int = 0

    @property
    def games_as_x(self):
        return self.x_wins + self.x_draws + self.x_losses

    @property
    def games_as_o(self):
        return self.o_wins + self.o_draws + self.o_losses

    @property
    def wins(self):
        return self.x_wins + self.o_wins

    @property
    def draws(self):
        return self.x_draws + self.o_draws

    @property
    def losses(self):
        return self.x_losses + self.o_losses

    @property
    def total_games(self) -> int:
        return self.games_as_x + self.games_as_o

    def score(self, xo="both"):
        wins = self.x_wins if xo == "x" else self.o_wins if xo == "o" else self.wins
        losses = (
            self.x_losses if xo == "x" else self.o_losses if xo == "o" else self.losses
        )
        return (wins - losses) / self.total_games


def full_round_robin(agents: list[tuple[str, Agent]], games_per_matchup: int):
    stats = [
        AgentStats(
            agent=agent,
            name=name,
        )
        for name, agent in agents
    ]
    for i in range(len(stats) - 1):
        for j in range(i + 1, len(stats)):
            a1, a2 = stats[i], stats[j]
            games_per_side = games_per_matchup // 2

            w, ll, d = play_games_parallel(a1.agent, a2.agent, games_per_side)
            a1.x_wins += w
            a1.x_losses += ll
            a1.x_draws += d
            a2.o_wins += ll
            a2.o_losses += w
            a2.o_draws += d

            w, ll, d = play_games_parallel(a2.agent, a1.agent, games_per_side)
            a2.x_wins += w
            a2.x_losses += ll
            a2.x_draws += d
            a1.o_wins += ll
            a1.o_losses += w
            a1.o_draws += d
    return stats


def print_rankings(stats: list[AgentStats]):
    print("\n" + "=" * 120)
    print("AGENT RANKINGS")
    print("=" * 120)
    print(
        f"{'Rank':<6} {'Name':<20} {'Score':<10} {'X Score':<10} {'O Score':<10} "
        f"{'Overall W-L-D':<18} {'X W-L-D':<18} {'O W-L-D':<18}"
    )
    print("-" * 120)

    for i, rating in enumerate(sorted(stats, key=lambda a: a.score(), reverse=True), 1):
        overall_wld = f"{rating.wins}-{rating.losses}-{rating.draws}"
        x_wld = f"{rating.x_wins}-{rating.x_losses}-{rating.x_draws}"
        o_wld = f"{rating.o_wins}-{rating.o_losses}-{rating.o_draws}"
        print(
            f"{i:<6} {rating.name:<20} {rating.score():<10.3f} {rating.score('x'):<10.3f} {rating.score('o'):<10.3f} "
            f"{overall_wld:<18} {x_wld:<18} {o_wld:<18}"
        )

    print("=" * 120 + "\n")
