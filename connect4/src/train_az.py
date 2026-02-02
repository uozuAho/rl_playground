import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

from torch.optim import Adam
import matplotlib.pyplot as plt

from agents import mcts_agent
from agents.agent import Agent
import agents.alphazero as az
from agents.alphazero import make_az_agent
from agents.az_nets import ResNet, AzNet
from agents.simple import RandomAgent, FirstLegalActionAgent
from utils.play import play_games_parallel


PROJ_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJ_ROOT/"experiments"


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
    experiment_description: str | None


@dataclass
class EvalConfig:
    n_games: int
    n_mcts_sims: int
    opponents: list[tuple[str, Agent]]


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
        minlen = min(len(x) for x in (self.games_played, self.value_losses, self.policy_losses))
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
        n = n if n is not None else min(len(x) for x in (self.win_rates, self.loss_rates, self.draw_rates))
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
    num_res_blocks=1,
    num_hidden=1,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_iterations=1,
    n_mcts_sims=20,
    n_games_per_iteration=100,
    n_epochs_per_iteration=2,
    epoch_batch_size=64,
    mask_invalid_actions=False,
    experiment_description="",
)


profile_train_config = TrainConfig(
    num_res_blocks=1,
    num_hidden=1,
    learning_rate=0.001,
    weight_decay=0.0001,
    num_iterations=1,
    n_mcts_sims=20,
    n_games_per_iteration=10,
    n_epochs_per_iteration=1,
    epoch_batch_size=10,
    mask_invalid_actions=False,
    experiment_description="profile",
)

default_eval_config = EvalConfig(
    n_games=20,
    n_mcts_sims=10,
    opponents=[
        ("random", RandomAgent()),
        ("first legal", FirstLegalActionAgent()),
        ("mctsu10", mcts_agent.make_uniform_agent(10))
    ],
)

profile_eval_config = EvalConfig(
    n_games=10,
    n_mcts_sims=10,
    opponents=[("random", RandomAgent())],
)


def main(mode):
    print("mode:", mode)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    train_config = profile_train_config if mode == "profile" else default_train_config
    eval_config = profile_eval_config if mode == "profile" else default_eval_config
    # device = "cpu"
    device = "cuda"
    if mode == "profile":
        pprint(train_config)
        pprint(eval_config)
        print("device:", device)

    net = ResNet(
        num_res_blocks=train_config.num_res_blocks,
        num_hidden=train_config.num_hidden,
        device=device,
    )
    optimiser = Adam(
        net.model.parameters(),
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
                device=device,
            )
            train_metrics.add(tmetrics)
            emetrics = eval_net(net, eval_config, device)
            eval_metrics.add(emetrics)
            print_train_metrics(train_metrics)
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
    net: AzNet,
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


def eval_net(net: AzNet, config: EvalConfig, device):
    metrics = EvalMetrics()
    aza = make_az_agent(net, config.n_mcts_sims, device)
    start = time.perf_counter()
    for oname, oagent in config.opponents:
        w, ll, d = play_games_parallel(aza, oagent, config.n_games)
        mr = MatchResults(p1_name="az", p2_name=oname, p1_wins=w, p1_losses=ll, draws=d)
        metrics.win_rates[f"vs {oname}"].append(mr.p1_win_rate)
        metrics.loss_rates[f"vs {oname}"].append(mr.p1_loss_rate)
        metrics.draw_rates[f"vs {oname}"].append(mr.draw_rate)
    dur = time.perf_counter() - start
    metrics.games_played = config.n_games * len(config.opponents)
    metrics.total_play_time += dur
    return metrics


def print_train_metrics(metrics: TrainingMetrics):
    ngames = metrics.games_played[-1]
    ploss = metrics.policy_losses[-1]
    vloss = metrics.value_losses[-1]
    gsec = metrics.games_per_sec
    print(f"train: {ngames} games. {gsec:.2f} games/sec. pv loss {ploss:.3f} {vloss:.3f}")


def print_eval_metrics(metrics: EvalMetrics):
    namecol_w = 3 + max(len(k) for k in metrics.win_rates)
    print(f"eval: {metrics.games_per_sec:.2f} games/sec")
    for k in metrics.win_rates:
        w,ll,d = metrics.win_rates[k][-1], metrics.loss_rates[k][-1], metrics.draw_rates[k][-1]
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
    config_text += f"Res Blocks: {train_config.num_res_blocks}\n"
    config_text += f"Hidden Units: {train_config.num_hidden}\n"
    config_text += f"Batch Size: {train_config.epoch_batch_size}\n"
    config_text += f"Train MCTS Sims: {train_config.n_mcts_sims}\n"
    config_text += f"Eval MCTS Sims: {eval_config.n_mcts_sims}\n"
    config_text += f"Games/itr: {train_config.n_games_per_iteration}\n"
    config_text += f"Epochs/itr: {train_config.n_epochs_per_iteration}\n"
    config_text += f"LR: {train_config.learning_rate}\n"
    config_text += f"Weight Decay: {train_config.weight_decay}\n"
    config_text += f"Train games/s: {train_metrics.games_per_sec:.2f}"

    if train_config.experiment_description:
        config_text += f"\n\n{train_config.experiment_description}"

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
    plt.savefig(EXPERIMENTS_DIR/"train_az.png")
    plt.show()


if __name__ == "__main__":
    if "profile" in sys.argv:
        main("profile")
    else:
        main("normal")
