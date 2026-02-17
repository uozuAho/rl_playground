"""Same goal as train az, but use alphazero mp (multiprocessing), aiming for
max GPU usage and therefore fastest possible training"""

import json
import os.path
import queue
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

from matplotlib import pyplot as plt

import agents.alphazero_mp as az
from agents.alphazero_mp import LearnerMetrics, EvalMetrics, player_loop
from agents import mcts_agent

PROJ_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJ_ROOT / "train_az_mp.log"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"


def main(args: list[str]):
    config = az.Config(
        num_res_blocks=9,
        num_hidden=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        mask_invalid_actions=True,
        train_n_mcts_sims=60,
        train_c_puct=2.0,
        temperature=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        n_player_processes=2,
        player_n_parallel_games=40,
        epoch_size=512,
        n_epoch_repeats=4,
        batch_size=128,
        weights_update_interval=1,
        discard_on_weight_update=False,
        eval_opponents=[
            ("mctsrr20", mcts_agent.make_random_rollout_agent(n_sims=20)),
        ],
        device_player="cuda",
        device_learn="cuda",
        device_eval="cpu",
        stop_after_n_seconds=None,
        stop_after_n_learns=None,
        cli_log_mode="eval",
        log_to_file=True,
        log_format_file="json",
        console_log_level="INFO",
        log_file_path=LOG_PATH,
    )
    print("args:", args)
    if "profile_player" in args:
        player_loop(
            "player", queue.Queue(), queue.Queue(), queue.Queue(), mp.Event(), config
        )
    else:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        az.train_mp(config)
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
                        lm = LearnerMetrics(**metobj)
                        steps = lm["steps_trained"]
                        azmetrics.policy_loss[steps] = lm["policy_loss"]
                        azmetrics.value_loss[steps] = lm["value_loss"]
                    case "eval":
                        em = EvalMetrics(**metobj)
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


def plot_metrics(metrics: AzMetrics, train_config: az.Config):
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
    config_text += f"Res Blocks: {train_config.num_res_blocks}\n"
    config_text += f"Hidden Units: {train_config.num_hidden}\n"
    config_text += f"LR: {train_config.learning_rate}\n"
    config_text += f"Weight Decay: {train_config.weight_decay}\n"
    config_text += f"Train MCTS Sims: {train_config.train_n_mcts_sims}\n"
    config_text += f"Eval MCTS Sims: {train_config.eval_n_mcts_sims}\n"
    config_text += f"Games/itr: {train_config.epoch_size}\n"
    config_text += f"Epochs/itr: {train_config.n_epoch_repeats}\n"
    config_text += f"Batch Size: {train_config.batch_size}\n"
    config_text += f"Train mask: {train_config.mask_invalid_actions}\n"

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
    plt.savefig(EXPERIMENTS_DIR / "train_az_mp.png")
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
