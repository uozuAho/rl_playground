import json
from collections import defaultdict
from datetime import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib import pyplot as plt

import agents.alphazero.azmp as az
from agents.random import RandomAgent

PROJ_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJ_ROOT / "train_az.log"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"


def main():
    config = az.Config(
        num_res_blocks=2,
        num_hidden=24,
        learning_rate=0.001,
        weight_decay=0.0001,
        mask_invalid_actions=True,
        train_n_mcts_sims=4,
        train_c_puct=2.0,
        temperature=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        n_player_processes=1,
        player_n_parallel_games=1,
        epoch_size=512,
        n_epoch_repeats=4,
        batch_size=128,
        weights_update_interval=1,
        discard_on_weight_update=False,
        eval_opponents=[
            ("rng", RandomAgent()),
        ],
        eval_n_games=2,
        eval_n_mcts_sims=4,
        device_player="cuda",
        device_learn="cuda",
        device_eval="cuda",
        cli_log_mode="perf",
        log_to_file=True,
        log_format_file="json",
        console_log_level="INFO",
        log_file_path=LOG_PATH,
    )
    # if os.path.exists(LOG_PATH):
    #     os.remove(LOG_PATH)
    # az.train_mp(config)
    metrics = log2metrics(LOG_PATH)
    plot_metrics(metrics, config)


@dataclass
class AzMetrics:
    # policy & value loss vs steps
    policy_loss: dict[int, float] = field(default_factory=dict)
    value_loss: dict[int, float] = field(default_factory=dict)

    # WLD rates vs time, per agent matchup
    win_rates: dict[str, dict[datetime, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    loss_rates: dict[str, dict[datetime, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    draw_rates: dict[str, dict[datetime, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )


def log2metrics(log_path: Path):
    azmetrics = AzMetrics()
    with open(log_path) as infile:
        for line in infile:
            logobj = json.loads(line)
            timestamp = datetime.fromisoformat(logobj["timestamp"])
            if logobj["process"] == "metrics":
                metobj = json.loads(logobj["message"].replace("'", '"'))
                match metobj["type"]:
                    case "learner":
                        lm = az.LearnerMetrics(**metobj)
                        steps = lm["steps_trained"]
                        azmetrics.policy_loss[steps] = lm["policy_loss"]
                        azmetrics.value_loss[steps] = lm["value_loss"]
                    case "eval":
                        em = az.EvalMetrics(**metobj)
                        for name, rate in em["win_rates"].items():
                            azmetrics.win_rates[name][timestamp] = rate
                        for name, rate in em["loss_rates"].items():
                            azmetrics.loss_rates[name][timestamp] = rate
    return azmetrics


def as_xy(d: dict) -> tuple[list, list]:
    return tuple(zip(*sorted(d.items())))


def to_relative_time(t: list[datetime]):
    start = t[0]
    return [(x - start).total_seconds() for x in t]


def plot_metrics(metrics: AzMetrics, config: az.Config):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = as_xy(metrics.policy_loss)
    axes[0, 0].plot(x, y, marker="o")
    axes[0, 0].set_xlabel("Steps trained")
    axes[0, 0].set_ylabel("Policy Loss")
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].grid(True)

    x, y = as_xy(metrics.value_loss)
    axes[0, 1].plot(x, y, marker="o")
    axes[0, 1].set_xlabel("Steps trained")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].grid(True)

    for k in metrics.win_rates:
        x, y = as_xy(metrics.win_rates[k])
        x = to_relative_time(x)
        axes[1, 0].plot(x, y, marker="o", label=k)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_title("Win Rates")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    for k in metrics.loss_rates:
        x, y = as_xy(metrics.loss_rates[k])
        x = to_relative_time(x)
        axes[1, 1].plot(x, y, marker="o", label=k)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Loss Rate")
    axes[1, 1].set_title("Loss Rates")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    config_text = "Configuration:\n"
    config_text += f"Res Blocks: {config.num_res_blocks}\n"
    config_text += f"Hidden Units: {config.num_hidden}\n"
    config_text += f"LR: {config.learning_rate}\n"
    config_text += f"Weight Decay: {config.weight_decay}\n"
    config_text += f"Train MCTS Sims: {config.train_n_mcts_sims}\n"
    config_text += f"Eval MCTS Sims: {config.eval_n_mcts_sims}\n"
    config_text += f"Games/itr: {config.epoch_size}\n"
    config_text += f"Epochs/itr: {config.n_epoch_repeats}\n"
    config_text += f"Batch Size: {config.batch_size}\n"
    config_text += f"Train mask: {config.mask_invalid_actions}\n"
    config_text += f"Discard: {config.discard_on_weight_update}\n"

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
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    plt.savefig(EXPERIMENTS_DIR / "train_az.png")
    plt.show()


if __name__ == "__main__":
    main()
