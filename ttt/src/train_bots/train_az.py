import time
from pathlib import Path

from torch.optim import Adam
import matplotlib.pyplot as plt

import ensure_project_path  # noqa  python sucks

import agents.alphazero as az
from agents.alphazero import ResNet
from agents.compare import play_and_report, play_games_parallel
from agents.mcts import MctsAgent
from agents.perfect import PerfectAgent
from agents.random import RandomAgent


PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    num_iterations = 502345
    experiment_description = ""
    num_res_blocks = 4
    num_hidden = 64
    learning_rate = 0.001
    weight_decay = 0.0001
    # device = "cpu"
    device = "cuda"
    saved_model_path = PROJECT_ROOT / "trained_models/az_ttt"

    model = az.ResNet(
        num_res_blocks=num_res_blocks, num_hidden=num_hidden, device=device
    )
    optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    try:
        train(
            device,
            model,
            num_hidden,
            num_res_blocks,
            optimiser,
            num_iterations=num_iterations,
            n_mcts_sims_train=60,
            n_games_per_nn_update=200,
            n_epochs_per_nn_update=4,
            epoch_batch_size=64,
            mask_invalid_actions=False,
            experiment_description=experiment_description,
        )
        load_and_play(device, saved_model_path)
    except KeyboardInterrupt:
        pass
        # print("saving & stopping...")
        # az.save(model, num_res_blocks, num_hidden, saved_model_path)


def load_and_play(device: str, saved_model_path: Path):
    loaded_az = az.load_net(saved_model_path, device)
    aza = az.AlphaZeroAgent.from_nn(model=loaded_az, device=device, n_mcts_sims=20)
    rng = RandomAgent()
    print("playing with loaded net...")
    play_and_report(aza, "az", rng, "rng", 20)


def train(
    device: str,
    model: ResNet,
    num_hidden: int,
    num_res_blocks: int,
    optimiser: Adam,
    num_iterations: int,
    n_mcts_sims_train: int,
    n_games_per_nn_update: int,
    n_epochs_per_nn_update: int,
    epoch_batch_size: int,
    mask_invalid_actions: bool,
    experiment_description="",
):
    """
    Params:
    - num_iterations: 1 iteration = play n_games_per_nn_update games, update the net, eval vs opponent
    """
    games_played = []
    policy_losses = []
    value_losses = []
    win_rates = []
    loss_rates = []
    win_rates_m10rr = []
    loss_rates_m10rr = []
    win_rates_perfect = []
    loss_rates_perfect = []

    config = {
        "num_res_blocks": num_res_blocks,
        "num_hidden": num_hidden,
        "batch_size": epoch_batch_size,
        "n_mcts_sims": n_mcts_sims_train,
        "n_games_per_update": n_games_per_nn_update,
        "n_epochs_per_update": n_epochs_per_nn_update,
        "learning_rate": optimiser.param_groups[0]["lr"],
        "weight_decay": optimiser.param_groups[0]["weight_decay"],
    }

    try:
        for iteration in range(num_iterations):
            iter_policy_losses = []
            iter_value_losses = []

            start = time.perf_counter()
            pl, vl = az.train(
                model,
                optimiser,
                n_games=n_games_per_nn_update,
                n_epochs=n_epochs_per_nn_update,
                n_mcts_sims=n_mcts_sims_train,
                device=device,
                train_batch_size=epoch_batch_size,
                mask_invalid_actions=mask_invalid_actions,
                verbose=False,
                parallel=True,
            )
            end = time.perf_counter()
            dur = end - start
            print(f"train: {n_games_per_nn_update / dur} games/sec")
            iter_policy_losses.append(pl)
            iter_value_losses.append(vl)

            games_played.append((iteration + 1) * n_games_per_nn_update)
            avg_pl = sum(iter_policy_losses) / len(iter_policy_losses)
            avg_vl = sum(iter_value_losses) / len(iter_value_losses)
            policy_losses.append(avg_pl)
            value_losses.append(avg_vl)

            aza = az.AlphaZeroAgent.from_nn(model, device=device, n_mcts_sims=20)
            rng = RandomAgent()
            m10rr = MctsAgent(n_sims=10)
            perfect = PerfectAgent()
            n_eval_games = 20

            start = time.perf_counter()
            results_rng = play_games_parallel(aza, rng, n_eval_games)
            win_rate = results_rng["X"] / n_eval_games
            loss_rate = results_rng["O"] / n_eval_games
            win_rates.append(win_rate)
            loss_rates.append(loss_rate)

            results_m10rr = play_games_parallel(aza, m10rr, n_eval_games)
            win_rate_m10rr = results_m10rr["X"] / n_eval_games
            loss_rate_m10rr = results_m10rr["O"] / n_eval_games
            win_rates_m10rr.append(win_rate_m10rr)
            loss_rates_m10rr.append(loss_rate_m10rr)

            results_perfect = play_games_parallel(aza, perfect, n_eval_games)
            w, ll, d = (
                results_perfect["X"],
                results_perfect["O"],
                results_perfect["draw"],
            )
            print("perfect WLD: ", w, ll, d)
            win_rate_perfect = results_perfect["X"] / n_eval_games
            loss_rate_perfect = results_perfect["O"] / n_eval_games
            win_rates_perfect.append(win_rate_perfect)
            loss_rates_perfect.append(loss_rate_perfect)
            end = time.perf_counter()
            dur = end - start
            print(f"eval: {n_eval_games * 3 / dur} games/sec")

            print(
                f"g {games_played[-1]}: policy_loss={avg_pl:.4f}, value_loss={avg_vl:.4f}, "
                f"vs_rng: wr={win_rate:.2%} lr={loss_rate:.2%}, "
                f"vs_m10rr: wr={win_rate_m10rr:.2%} lr={loss_rate_m10rr:.2%}, "
                f"vs_perfect: wr={win_rate_perfect:.2%} lr={loss_rate_perfect:.2%}"
            )

    except KeyboardInterrupt:
        minlen = min(
            len(x)
            for x in (
                games_played,
                policy_losses,
                value_losses,
                win_rates,
                loss_rates,
                win_rates_m10rr,
                loss_rates_m10rr,
                win_rates_perfect,
                loss_rates_perfect,
            )
        )
        games_played = games_played[:minlen]
        policy_losses = policy_losses[:minlen]
        value_losses = value_losses[:minlen]
        win_rates = win_rates[:minlen]
        loss_rates = loss_rates[:minlen]
        win_rates_m10rr = win_rates_m10rr[:minlen]
        loss_rates_m10rr = loss_rates_m10rr[:minlen]
        win_rates_perfect = win_rates_perfect[:minlen]
        loss_rates_perfect = loss_rates_perfect[:minlen]
        raise
    finally:
        plot_training_metrics(
            games_played,
            policy_losses,
            value_losses,
            win_rates,
            loss_rates,
            win_rates_m10rr,
            loss_rates_m10rr,
            win_rates_perfect,
            loss_rates_perfect,
            config,
            experiment_description,
        )


def plot_training_metrics(
    games_played,
    policy_losses,
    value_losses,
    win_rates,
    loss_rates,
    win_rates_m10rr,
    loss_rates_m10rr,
    win_rates_perfect,
    loss_rates_perfect,
    config,
    custom_text="bleep bloop",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(games_played, policy_losses, marker="o")
    axes[0, 0].set_xlabel("Games")
    axes[0, 0].set_ylabel("Policy Loss")
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(games_played, value_losses, marker="o")
    axes[0, 1].set_xlabel("Games")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].grid(True)

    axes[1, 0].plot(games_played, win_rates, marker="o", label="vs Random")
    axes[1, 0].plot(games_played, win_rates_m10rr, marker="s", label="vs m10rr")
    axes[1, 0].plot(games_played, win_rates_perfect, marker="^", label="vs Perfect")
    axes[1, 0].set_xlabel("Games")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].set_title("Win Rates")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(games_played, loss_rates, marker="o", label="vs Random")
    axes[1, 1].plot(games_played, loss_rates_m10rr, marker="s", label="vs m10rr")
    axes[1, 1].plot(games_played, loss_rates_perfect, marker="^", label="vs Perfect")
    axes[1, 1].set_xlabel("Games")
    axes[1, 1].set_ylabel("Loss Rate")
    axes[1, 1].set_title("Loss Rates")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    config_text = "Configuration:\n"
    config_text += f"Res Blocks: {config['num_res_blocks']}\n"
    config_text += f"Hidden Units: {config['num_hidden']}\n"
    config_text += f"Batch Size: {config['batch_size']}\n"
    config_text += f"MCTS Sims: {config['n_mcts_sims']}\n"
    config_text += f"Games/Update: {config['n_games_per_update']}\n"
    config_text += f"Epochs/Update: {config['n_epochs_per_update']}\n"
    config_text += f"LR: {config['learning_rate']}\n"
    config_text += f"Weight Decay: {config['weight_decay']}"

    if custom_text:
        config_text += f"\n\n{custom_text}"

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
    (PROJECT_ROOT / "experiment_logs").mkdir(exist_ok=True)
    plt.savefig(PROJECT_ROOT / "experiment_logs/train_az.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
