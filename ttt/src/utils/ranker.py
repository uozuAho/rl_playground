from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

from ttt.env import TttEnv, X, O, Player, Status, DRAW
from agents.agent import TttAgent


@dataclass
class AgentStats:
    agent: TttAgent
    name: str
    games_as_x: int = 0
    games_as_o: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    x_wins: int = 0
    x_losses: int = 0
    x_draws: int = 0
    o_wins: int = 0
    o_losses: int = 0
    o_draws: int = 0

    @property
    def total_games(self) -> int:
        return self.games_as_x + self.games_as_o

    def score(self, xo="both"):
        wins = self.x_wins if xo == "x" else self.o_wins if xo == "o" else self.wins
        losses = (
            self.x_losses if xo == "x" else self.o_losses if xo == "o" else self.losses
        )
        return (wins - losses) / self.total_games


def play_game(agent1: TttAgent, agent2: TttAgent, agent1_player: Player) -> Status:
    env = TttEnv()

    while True:
        current_agent = agent1 if env.current_player == agent1_player else agent2
        action = current_agent.get_action(env)

        _, reward, done, _, _ = env.step(action)

        if done:
            return env.status()


class AgentRanker:
    def __init__(
        self,
        agents: list[tuple[str, TttAgent]],
    ):
        self.agents = [
            AgentStats(
                agent=agent,
                name=name,
            )
            for name, agent in agents
        ]

    def full_round_robin(self, games_per_matchup: int):
        for i in range(len(self.agents) - 1):
            for j in range(i + 1, len(self.agents)):
                a1, a2 = self.agents[i], self.agents[j]
                for g in range(games_per_matchup):
                    a1_player = X if g < games_per_matchup / 2 else O
                    result = play_game(a1.agent, a2.agent, a1_player)
                    self._update_stats(a1, a1_player, a2, result)
        return self.agents

    def _update_stats(
        self,
        agent1: AgentStats,
        agent1_player: int | Any,
        agent2: AgentStats,
        result: Literal[-1, 1, 2, 3],
    ):
        if agent1_player == X:
            agent1.games_as_x += 1
            agent2.games_as_o += 1
        else:
            agent1.games_as_o += 1
            agent2.games_as_x += 1

        if result == DRAW:
            agent1.draws += 1
            agent2.draws += 1
            if agent1_player == X:
                agent1.x_draws += 1
                agent2.o_draws += 1
            else:
                agent1.o_draws += 1
                agent2.x_draws += 1
        elif result == agent1_player:
            agent1.wins += 1
            agent2.losses += 1
            if agent1_player == X:
                agent1.x_wins += 1
                agent2.o_losses += 1
            else:
                agent1.o_wins += 1
                agent2.x_losses += 1
        else:
            agent1.losses += 1
            agent2.wins += 1
            if agent1_player == X:
                agent1.x_losses += 1
                agent2.o_wins += 1
            else:
                agent1.o_losses += 1
                agent2.x_wins += 1

    def print_rankings(self, stats: list[AgentStats]):
        print("\n" + "=" * 120)
        print("AGENT RANKINGS")
        print("=" * 120)
        print(
            f"{'Rank':<6} {'Name':<20} {'Score':<10} {'X Score':<10} {'O Score':<10} "
            f"{'Overall W-L-D':<18} {'X W-L-D':<18} {'O W-L-D':<18}"
        )
        print("-" * 120)

        for i, rating in enumerate(
            sorted(stats, key=lambda a: a.score(), reverse=True), 1
        ):
            overall_wld = f"{rating.wins}-{rating.losses}-{rating.draws}"
            x_wld = f"{rating.x_wins}-{rating.x_losses}-{rating.x_draws}"
            o_wld = f"{rating.o_wins}-{rating.o_losses}-{rating.o_draws}"
            print(
                f"{i:<6} {rating.name:<20} {rating.score():<10.3f} {rating.score('x'):<10.3f} {rating.score('o'):<10.3f} "
                f"{overall_wld:<18} {x_wld:<18} {o_wld:<18}"
            )

        print("=" * 120 + "\n")
