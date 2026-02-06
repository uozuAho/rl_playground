import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

from torch.optim import Adam
import matplotlib.pyplot as plt

import agents.alphazero as az
from agents.agent import TttAgent
from agents.alphazero import AlphaZeroAgent
from agents.az_nets import ResNet
from agents.compare import play_games_parallel
from agents.mcts import MctsAgent
from agents.perfect import PerfectAgent
from agents.random import RandomAgent

PROJ_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJ_ROOT / "experiment_logs"


@dataclass
class TrainConfig:
    num_res_blocks: int
    num_hidden: int
    learning_rate: float
    weight_decay: float
    num_iterations: int  # iteration: play games, update net
    n_mcts_sims: int
    n_games_per_iteration: int
    n_epochs_per_iteration: int
    epoch_batch_size: int
    mask_invalid_actions: bool
    device: str


@dataclass
class EvalConfig:
    n_games: int
    n_mcts_sims: int
    opponents: list[tuple[str, TttAgent]]


@dataclass
class TrainingMetrics:
    games_played: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    total_training_time: float = 0

    @property
    def total_games_played(self):
        return self.games_played[-1]

    @property
    def games_per_sec(self):
        return self.total_games_played / self.total_training_time

    def add(self, metrics: "TrainingMetrics"):
        n_games = 0 if len(self.games_played) == 0 else self.games_played[-1]
        self.games_played.extend(x + n_games for x in metrics.games_played)
        self.policy_losses.extend(metrics.policy_losses)
        self.value_losses.extend(metrics.value_losses)
        self.total_training_time += metrics.total_training_time

    def trim(self):
        minlen = min(
            len(x) for x in (self.games_played, self.value_losses, self.policy_losses)
        )
        self.games_played = self.games_played[:minlen]
        self.policy_losses = self.policy_losses[:minlen]
        self.value_losses = self.value_losses[:minlen]
        return minlen


@dataclass
class EvalMetrics:
    # dict["vs opponent": [rate per evaluation]]
    win_rates: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    loss_rates: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    draw_rates: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    games_played: int = 0
    total_play_time: float = 0

    @property
    def games_per_sec(self):
        return self.games_played / self.total_play_time

    def add(self, metrics: "EvalMetrics"):
        for k in metrics.win_rates:
            self.win_rates[k].extend(metrics.win_rates[k])
        for k in metrics.loss_rates:
            self.loss_rates[k].extend(metrics.loss_rates[k])
        for k in metrics.draw_rates:
            self.draw_rates[k].extend(metrics.draw_rates[k])
        self.games_played += metrics.games_played
        self.total_play_time += metrics.total_play_time

    def trim(self, n: int | None):
        n = (
            n
            if n is not None
            else min(len(x) for x in (self.win_rates, self.loss_rates, self.draw_rates))
        )
        for k in self.win_rates:
            self.win_rates[k] = self.win_rates[k][:n]
        for k in self.loss_rates:
            self.loss_rates[k] = self.loss_rates[k][:n]
        for k in self.draw_rates:
            self.draw_rates[k] = self.draw_rates[k][:n]


@dataclass
class MatchResults:
    p1_name: str
    p2_name: str
    p1_wins: int
    p1_losses: int
    draws: int

    @property
    def games_played(self):
        return self.p1_wins + self.p1_losses + self.draws

    @property
    def p1_win_rate(self):
        return self.p1_wins / self.games_played

    @property
    def p1_loss_rate(self):
        return self.p1_losses / self.games_played

    @property
    def draw_rate(self):
        return self.draws / self.games_played


default_train_config = TrainConfig(
    num_res_blocks=4,
    num_hidden=64,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_iterations=1,
    n_mcts_sims=40,
    n_games_per_iteration=50,
    n_epochs_per_iteration=4,
    epoch_batch_size=128,
    mask_invalid_actions=True,
    # device="cpu",
    device="cuda",
)


default_eval_config = EvalConfig(
    n_games=20,
    n_mcts_sims=10,
    opponents=[
        ("random", RandomAgent()),
        ("perfect", PerfectAgent()),
        ("mcts_rr10", MctsAgent(n_sims=10)),
    ],
)


def main(mode):
    print("mode:", mode)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    train_config = default_train_config
    eval_config = default_eval_config
    if mode == "profile":
        pprint(train_config)
        pprint(eval_config)
        print("device:", train_config.device)

    net = ResNet(
        num_res_blocks=train_config.num_res_blocks,
        num_hidden=train_config.num_hidden,
        device=train_config.device,
    )
    optimiser = Adam(
        net.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    train_metrics = TrainingMetrics()
    eval_metrics = EvalMetrics()
    start = time.perf_counter()
    try:
        while True:
            tmetrics = train(
                net,
                optimiser,
                train_config,
                device=train_config.device,
            )
            train_metrics.add(tmetrics)
            print_train_metrics(train_metrics)
            emetrics = eval_net(net, eval_config, train_config.device)
            eval_metrics.add(emetrics)
            print_eval_metrics(eval_metrics)
            if mode == "profile" and time.perf_counter() - start > 5:
                break
    except KeyboardInterrupt:
        if mode != "profile":
            n = train_metrics.trim()
            eval_metrics.trim(n)
            plot_training_metrics(
                train_config, eval_config, train_metrics, eval_metrics
            )


def train(
    net: ResNet,
    optimiser: Adam,
    train_config: TrainConfig,
    device: str,
):
    train_metrics = TrainingMetrics()

    for iteration in range(train_config.num_iterations):
        iter_policy_losses = []
        iter_value_losses = []

        start = time.perf_counter()
        pl, vl = az.train(
            net,
            optimiser,
            n_games=train_config.n_games_per_iteration,
            n_epochs=train_config.n_epochs_per_iteration,
            n_mcts_sims=train_config.n_mcts_sims,
            device=device,
            train_batch_size=train_config.epoch_batch_size,
            mask_invalid_actions=train_config.mask_invalid_actions,
            verbose=False,
        )
        end = time.perf_counter()
        dur = end - start
        train_metrics.total_training_time += dur
        iter_policy_losses.append(pl)
        iter_value_losses.append(vl)

        train_metrics.games_played.append(
            (iteration + 1) * train_config.n_games_per_iteration
        )
        avg_pl = sum(iter_policy_losses) / len(iter_policy_losses)
        avg_vl = sum(iter_value_losses) / len(iter_value_losses)
        train_metrics.policy_losses.append(avg_pl)
        train_metrics.value_losses.append(avg_vl)

    return train_metrics


def eval_net(net: ResNet, config: EvalConfig, device):
    metrics = EvalMetrics()
    aza = AlphaZeroAgent.from_nn(net, config.n_mcts_sims, device)
    start = time.perf_counter()
    for oname, oagent in config.opponents:
        for azplayer in ["x", "o"]:
            players = (aza, oagent) if azplayer == "x" else (oagent, aza)
            pnames = ("az", oname) if azplayer == "x" else (oname, "az")
            r = play_games_parallel(players[0], players[1], config.n_games)
            w,ll,d = r["X"], r["O"], r["draw"]
            mr = MatchResults(
                p1_name=pnames[0], p2_name=pnames[1], p1_wins=w, p1_losses=ll, draws=d
            )
            metrics.win_rates[f"{pnames[0]} vs {pnames[1]}"].append(mr.p1_win_rate)
            metrics.loss_rates[f"{pnames[0]} vs {pnames[1]}"].append(mr.p1_loss_rate)
            metrics.draw_rates[f"{pnames[0]} vs {pnames[1]}"].append(mr.draw_rate)
    dur = time.perf_counter() - start
    metrics.games_played = config.n_games * len(config.opponents)
    metrics.total_play_time += dur
    return metrics


def print_train_metrics(metrics: TrainingMetrics):
    ngames = metrics.games_played[-1]
    ploss = metrics.policy_losses[-1]
    vloss = metrics.value_losses[-1]
    gsec = metrics.games_per_sec
    print(
        f"train: {ngames} games. {gsec:.2f} games/sec. pv loss {ploss:.3f} {vloss:.3f}"
    )


def print_eval_metrics(metrics: EvalMetrics):
    namecol_w = 3 + max(len(k) for k in metrics.win_rates)
    print(f"eval: {metrics.games_per_sec:.2f} games/sec")
    for k in metrics.win_rates:
        w, ll, d = (
            metrics.win_rates[k][-1],
            metrics.loss_rates[k][-1],
            metrics.draw_rates[k][-1],
        )
        print(f"{k.ljust(namecol_w)} WLD: {w:5.2f} {ll:5.2f} {d:5.2f}")


def plot_training_metrics(
    train_config: TrainConfig,
    eval_config: EvalConfig,
    train_metrics: TrainingMetrics,
    eval_metrics: EvalMetrics,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(train_metrics.games_played, train_metrics.policy_losses, marker="o")
    axes[0, 0].set_xlabel("Games")
    axes[0, 0].set_ylabel("Policy Loss")
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(train_metrics.games_played, train_metrics.value_losses, marker="o")
    axes[0, 1].set_xlabel("Games")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].grid(True)

    for k in eval_metrics.win_rates:
        axes[1, 0].plot(
            train_metrics.games_played, eval_metrics.win_rates[k], marker="o", label=k
        )
    axes[1, 0].set_xlabel("Games")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_title("Win Rates")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for k in eval_metrics.loss_rates:
        axes[1, 1].plot(
            train_metrics.games_played, eval_metrics.loss_rates[k], marker="o", label=k
        )
    axes[1, 1].set_xlabel("Games")
    axes[1, 1].set_ylabel("Loss Rate")
    axes[1, 1].set_title("Loss Rates")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    config_text = "Configuration:\n"
    config_text += f"device: {train_config.device}\n"
    config_text += f"Res Blocks: {train_config.num_res_blocks}\n"
    config_text += f"Hidden Units: {train_config.num_hidden}\n"
    config_text += f"LR: {train_config.learning_rate}\n"
    config_text += f"Weight Decay: {train_config.weight_decay}\n"
    config_text += f"Train MCTS Sims: {train_config.n_mcts_sims}\n"
    config_text += f"Eval MCTS Sims: {eval_config.n_mcts_sims}\n"
    config_text += f"Games/itr: {train_config.n_games_per_iteration}\n"
    config_text += f"Epochs/itr: {train_config.n_epochs_per_iteration}\n"
    config_text += f"Batch Size: {train_config.epoch_batch_size}\n"
    config_text += f"Train mask: {train_config.mask_invalid_actions}\n"
    config_text += f"Train games/s: {train_metrics.games_per_sec:.2f}"

    fig.text(
        0.02,
        0.02,
        config_text,
        fontsize=8,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))  # Leave space for text box
    plt.savefig(EXPERIMENTS_DIR / "train_az.png")
    plt.show()


if __name__ == "__main__":
    if "profile" in sys.argv:
        main("profile")
    else:
        main("normal")
