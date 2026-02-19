from __future__ import annotations
from dataclasses import dataclass

from agents.agent import ChessAgent
from utils.play import play_games_parallel


@dataclass
class AgentStats:
    agent: ChessAgent
    name: str
    w_wins: int = 0
    w_losses: int = 0
    w_draws: int = 0
    b_wins: int = 0
    b_losses: int = 0
    b_draws: int = 0

    @property
    def games_as_x(self):
        return self.w_wins + self.w_draws + self.w_losses

    @property
    def games_as_o(self):
        return self.b_wins + self.b_draws + self.b_losses

    @property
    def wins(self):
        return self.w_wins + self.b_wins

    @property
    def draws(self):
        return self.w_draws + self.b_draws

    @property
    def losses(self):
        return self.w_losses + self.b_losses

    @property
    def total_games(self) -> int:
        return self.games_as_x + self.games_as_o

    def score(self, wb="both"):
        wins = self.w_wins if wb == "w" else self.b_wins if wb == "b" else self.wins
        losses = (
            self.w_losses if wb == "w" else self.b_losses if wb == "b" else self.losses
        )
        return (wins - losses) / self.total_games


def full_round_robin(agents: list[tuple[str, ChessAgent]], games_per_matchup: int):
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
            a1.w_wins += w
            a1.w_losses += ll
            a1.w_draws += d
            a2.b_wins += ll
            a2.b_losses += w
            a2.b_draws += d

            w, ll, d = play_games_parallel(a2.agent, a1.agent, games_per_side)
            a2.w_wins += w
            a2.w_losses += ll
            a2.w_draws += d
            a1.b_wins += ll
            a1.b_losses += w
            a1.b_draws += d
    return stats


def print_rankings(stats: list[AgentStats]):
    print("\n" + "=" * 120)
    print("AGENT RANKINGS")
    print("=" * 120)
    print(
        f"{'Rank':<6} {'Name':<20} {'Score':<10} {'W Score':<10} {'B Score':<10} "
        f"{'Overall W-L-D':<18} {'W W-L-D':<18} {'B W-L-D':<18}"
    )
    print("-" * 120)

    for i, rating in enumerate(sorted(stats, key=lambda a: a.score(), reverse=True), 1):
        overall_wld = f"{rating.wins}-{rating.losses}-{rating.draws}"
        w_wld = f"{rating.w_wins}-{rating.w_losses}-{rating.w_draws}"
        b_wld = f"{rating.b_wins}-{rating.b_losses}-{rating.b_draws}"
        print(
            f"{i:<6} {rating.name:<20} {rating.score():<10.3f} {rating.score('x'):<10.3f} {rating.score('o'):<10.3f} "
            f"{overall_wld:<18} {w_wld:<18} {b_wld:<18}"
        )

    print("=" * 120 + "\n")
