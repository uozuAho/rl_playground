"""Do stuff with pretrained tabular state-values:

- train the AZ NN against tabular data, optionally simulating AZ training with noise
- eval the NN against training data
- eval the AZ agent using the NN
"""

from dataclasses import dataclass, field
from typing import Literal

import ensure_project_path  # noqa

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from agents.agent import TttAgent

import ttt.env as t3
import agents.alphazero as az
from agents.alphazero import ResNet, GameStep
from agents.compare import play_and_report, play_games
from agents.random import RandomAgent
from agents.tab_greedy_v import TabGreedyVAgent
from utils import qtable, maths
from utils.maths import add_dirichlet_noise, kl_divergence

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    # device = "cpu"
    device = "cuda"
    saved_model_path = PROJECT_ROOT / "trained_models/az_tabnet"
    training_data = load_training_data(
        PROJECT_ROOT / "trained_models/tmcts_sym_100k_30"
    )

    train_config = TrainingConfig(
        num_iterations=200,
        num_res_blocks=5,
        num_hidden=64,
        learning_rate=0.001,
        weight_decay=0.05,
        batch_size=256,
        emulate_az_training=False,
        az_alpha=0.3,
        az_epsilon=0.2,
        policy_loss="ce",
        value_loss="huber",
        max_loss_coeff=1,
    )

    eval_config = EvalConfig(
        n_mcts_sims=10, do_agent_eval=False, do_nn_eval=True, n_eval_games=20
    )

    rnet = az.ResNet(
        num_res_blocks=train_config.num_res_blocks,
        num_hidden=train_config.num_hidden,
        device=device,
    )
    optimiser = Adam(
        rnet.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    metrics = train_and_evaluate(
        rnet,
        optimiser,
        training_data,
        train_config,
        eval_config,
        device,
    )
    rnet.save(saved_model_path)
    plot_training_metrics(
        metrics,
        train_config,
        eval_config,
    )
    # load_and_play(device, saved_model_path)


@dataclass
class TrainingConfig:
    num_res_blocks: int
    num_hidden: int
    batch_size: int
    num_iterations: int
    learning_rate: float
    weight_decay: float
    emulate_az_training: bool
    az_alpha: float
    az_epsilon: float
    policy_loss: Literal["ce", "log sm kl"]
    value_loss: Literal["mse", "huber"]
    max_loss_coeff: float


@dataclass
class EvalConfig:
    do_agent_eval: bool  # evaluate agent by playing games
    n_mcts_sims: int  # n sims for eval agent
    n_eval_games: int
    do_nn_eval: bool  # evaluate NN against training data


@dataclass
class TrainingMetrics:
    # number of passes over the training set. X axis.
    epochs: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    max_p_kldivs: list[float] = field(default_factory=list)
    avg_p_kldivs: list[float] = field(default_factory=list)
    max_v_abserrs: list[float] = field(default_factory=list)
    avg_v_abserrs: list[float] = field(default_factory=list)
    win_rates: list[float] = field(default_factory=list)
    loss_rates: list[float] = field(default_factory=list)

    def trim(self):
        """Ensure all metrics have the same length"""
        minlen = self.minlen()
        self.epochs = self.epochs[:minlen]
        self.policy_losses = self.policy_losses[:minlen]
        self.value_losses = self.value_losses[:minlen]
        self.max_p_kldivs = self.max_p_kldivs[:minlen]
        self.avg_p_kldivs = self.avg_p_kldivs[:minlen]
        self.max_v_abserrs = self.max_v_abserrs[:minlen]
        self.avg_v_abserrs = self.avg_v_abserrs[:minlen]
        self.win_rates = self.win_rates[:minlen]
        self.loss_rates = self.loss_rates[:minlen]

    def minlen(self):
        return min(
            len(x) if x else 9999999999999999999
            for x in (
                self.epochs,
                self.policy_losses,
                self.value_losses,
                self.max_p_kldivs,
                self.avg_p_kldivs,
                self.max_v_abserrs,
                self.avg_v_abserrs,
                self.win_rates,
                self.loss_rates,
            )
        )


def load_and_play(device: str, saved_model_path: Path):
    nn = ResNet.load(saved_model_path, device)
    aza = az.AlphaZeroAgent.from_nn(nn, n_mcts_sims=20, device=device)
    rng = RandomAgent()
    print("playing with loaded net...")
    play_and_report(aza, "az", rng, "rng", 20)


def load_training_data(path: Path, include_terminal=False) -> list[GameStep]:
    table = qtable.StateValueTable.load(path)
    states = list(extract_all_state_values(table))
    if not include_terminal:
        states = [s for s in states if t3.status(s[1]) == t3.IN_PROGRESS]
    return [
        GameStep(
            board,
            cp,
            valid_mask(board),
            qtable.greedy_probs(table, t3.TttEnv.from_str(bstr)),
            value,
        )
        for bstr, board, cp, value in states
    ]


def avg(x):
    return sum(x) / len(x)


def train_and_evaluate(
    rnet: ResNet,
    optimiser: Adam,
    game_steps: list[GameStep],
    train_config: TrainingConfig,
    eval_config: EvalConfig,
    device: str,
):
    """Train and optionally evaluate in a loop, recording metrics"""
    metrics = TrainingMetrics()

    def do_iteration(iteration):
        pl, vl = train_epoch(
            rnet=rnet,
            optimiser=optimiser,
            game_steps=game_steps,
            train_config=train_config,
            device=device,
        )

        metrics.epochs.append(iteration)
        metrics.policy_losses.append(pl)
        metrics.value_losses.append(vl)

        log_msg = f"{iteration}: policy loss={pl:.4f}, value loss={vl:.4f}"

        if eval_config.do_nn_eval:
            nnmetrics = eval_nn(device, game_steps, rnet)
            metrics.max_p_kldivs.append(max(nnmetrics.p_kldivs))
            metrics.avg_p_kldivs.append(avg(nnmetrics.p_kldivs))
            metrics.max_v_abserrs.append(max(nnmetrics.v_abserrs))
            metrics.avg_v_abserrs.append(avg(nnmetrics.v_abserrs))
            log_msg += f" | max errors: p: {max(nnmetrics.p_kldivs):0.2f}, v: {max(nnmetrics.v_abserrs):0.2f}"

        if eval_config.do_agent_eval:
            loss_rate, win_rate = eval_agent(
                device,
                eval_config.n_eval_games,
                eval_config.n_mcts_sims,
                rnet,
                RandomAgent(),
            )
            metrics.win_rates.append(win_rate)
            metrics.loss_rates.append(loss_rate)

            log_msg += f" | win_rate={win_rate:.2%}, loss_rate={loss_rate:.2%}"

        print(log_msg)

    for i in range(train_config.num_iterations):
        try:
            do_iteration(i)
        except KeyboardInterrupt:
            metrics.trim()
            break

    return metrics


def train_epoch(
    rnet: ResNet,
    optimiser: Adam,
    game_steps: list[GameStep],
    train_config: TrainingConfig,
    device: str,
):
    """Train for multiple epochs and return average losses"""
    rnet.train()
    policy_losses = []
    value_losses = []

    for _ in range(len(game_steps) // train_config.batch_size):
        sample = random.sample(game_steps, train_config.batch_size)
        if train_config.emulate_az_training:
            sample = [
                to_montecarlo_sample(s, train_config.az_alpha, train_config.az_epsilon)
                for s in sample
            ]
        pl, vl = az._update_net(
            rnet,
            optimiser,
            sample,
            device=device,
            mask_invalid_actions=True,
            policy_loss_type=train_config.policy_loss,
            value_loss_type=train_config.value_loss,
            max_loss_coeff=train_config.max_loss_coeff,
        )
        policy_losses.append(pl)
        value_losses.append(vl)

    avg_pl = sum(policy_losses) / len(policy_losses)
    avg_vl = sum(value_losses) / len(value_losses)
    return avg_pl, avg_vl


@dataclass
class NnEvalMetrics:
    game_steps: list[GameStep]
    p_kldivs: list[float]
    v_abserrs: list[float]

    def filter(self, predfn):
        for i, gs in enumerate(self.game_steps):
            if predfn(gs):
                yield gs, self.p_kldivs[i], self.v_abserrs[i]


def eval_nn(device: str, game_steps: list[GameStep], rnet: ResNet) -> NnEvalMetrics:
    """Evaluate nn using given data.
    Returns [tuple(GameStep, PolicyLoss (KL divergence), ValueLoss (abs err))]
    """
    boards, players, _, probs, values = zip(*[s.as_tuple() for s in game_steps])
    with torch.no_grad():
        pol_ests, val_ests = az._net_fwd_board_batch(rnet, boards, players, device)
        pol_ests = [maths.softmax(p) for p in pol_ests.cpu().numpy()]
        val_ests = val_ests.cpu().numpy().squeeze()
        p_kldivs = [kl_divergence(p, q) for p, q in zip(probs, pol_ests)]
        v_abserrs = [abs(v - v_est) for v, v_est in zip(values, val_ests)]
    return NnEvalMetrics(game_steps, p_kldivs, v_abserrs)


def eval_agent(
    device: str, n_eval_games: int, n_mcts_sims: int, rnet: ResNet, opponent: TttAgent
) -> tuple[float, float]:
    aza = az.AlphaZeroAgent.from_nn(model=rnet, device=device, n_mcts_sims=n_mcts_sims)
    results = play_games(aza, opponent, n_eval_games)
    win_rate = results["X"] / n_eval_games
    loss_rate = results["O"] / n_eval_games
    return loss_rate, win_rate


def evaluate(rnet: ResNet, device: str, n_mcts_sims=10, n_games=50):
    agent = az.AlphaZeroAgent.from_nn(
        model=rnet, device=device, n_mcts_sims=n_mcts_sims
    )
    rng = RandomAgent()
    rnet.eval()
    with torch.no_grad():
        play_and_report(agent, "az", rng, "rng", n_games=n_games)


def extract_all_state_values(table: qtable.StateValueTable):
    for env, value in table.values():
        yield env.str1d(), env.board, env.current_player, value


def action_values(tab_agent: TabGreedyVAgent, board_str: str):
    av = tab_agent.action_values(board_str)
    return [av.get(i, 0) for i in range(9)]


def valid_mask(board: t3.Board):
    valid_actions = list(t3.valid_actions(board))
    return [x in valid_actions for x in range(9)]


def to_montecarlo_sample(step: GameStep, alpha, epsilon):
    """Add noise to the action probabilities, and convert the final value to an end of game reward"""
    noisy_probs = add_dirichlet_noise(step.mcts_probs, alpha, epsilon)
    for i, v in enumerate(step.valid_action_mask):
        if not v:
            noisy_probs[i] = 0
    noisy_probs /= noisy_probs.sum()
    final_val = montecarlo_value(step.final_value)
    return GameStep(
        step.board, step.player, step.valid_action_mask, noisy_probs, final_val
    )


def montecarlo_value(v: float):
    """Simulate an MC episode value from a 'real' state value"""
    p1 = 0.33 + 0.67 * abs(v)
    p = [p1 / 2, p1 / 2, p1] if v >= 0 else [p1, p1 / 2, p1 / 2]
    p = [x / sum(p) for x in p]
    return np.random.choice([-1, 0, 1], p=p)


def plot_training_metrics(
    metrics: TrainingMetrics,
    train_config: TrainingConfig,
    eval_config: EvalConfig,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(
        metrics.epochs,
        metrics.policy_losses,
        marker="o",
        label="Training loss",
    )
    if eval_config.do_nn_eval:
        axes[0, 0].plot(
            metrics.epochs, metrics.max_p_kldivs, marker="s", label="Max eval KL div"
        )
        axes[0, 0].plot(
            metrics.epochs, metrics.avg_p_kldivs, marker="^", label="Avg eval KL div"
        )
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        metrics.epochs, metrics.value_losses, marker="o", label="Training loss"
    )
    if eval_config.do_nn_eval:
        axes[0, 1].plot(
            metrics.epochs, metrics.max_v_abserrs, marker="s", label="Eval max abs err"
        )
        axes[0, 1].plot(
            metrics.epochs, metrics.avg_v_abserrs, marker="^", label="Eval mean abs err"
        )
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if eval_config.do_agent_eval:
        axes[1, 0].plot(metrics.epochs, metrics.win_rates, marker="o")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Win Rate")
        axes[1, 0].set_title("Win Rate vs Random")
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True)

        axes[1, 1].plot(metrics.epochs, metrics.loss_rates, marker="o")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss Rate")
        axes[1, 1].set_title("Loss Rate vs Random")
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True)

    config_text = "Configuration:\n"
    config_text += f"Res Blocks: {train_config.num_res_blocks}\n"
    config_text += f"Hidden Units: {train_config.num_hidden}\n"
    config_text += f"Batch Size: {train_config.batch_size}\n"
    config_text += f"LR: {train_config.learning_rate}\n"
    config_text += f"Weight Decay: {train_config.weight_decay}\n"
    config_text += f"policy loss: {train_config.policy_loss}\n"
    config_text += f"value loss: {train_config.value_loss}\n"
    config_text += f"max loss coef: {train_config.max_loss_coeff}\n"
    config_text += f"Emulate AZ: {train_config.emulate_az_training}\n"
    if train_config.emulate_az_training:
        config_text += f"AZ Alpha: {train_config.az_alpha}\n"
        config_text += f"AZ Epsilon: {train_config.az_epsilon}\n"
    config_text += f"Eval n sims:{eval_config.n_mcts_sims}\n"

    fig.text(
        0.02,
        0.02,
        config_text,
        fontsize=8,
        family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(PROJECT_ROOT / "experiment_logs/az_tabnet.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
