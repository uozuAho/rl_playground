import time
from dataclasses import dataclass, field

from torch.optim import Adam
import matplotlib.pyplot as plt

from agents.agent import Agent
import agents.alphazero as az
from agents.alphazero import make_az_agent
from agents.az_nets import ResNet, AzNet
from agents.simple import RandomAgent
from utils.play import play_games_parallel
import env.connect4 as c4

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


def main():
    train_config = TrainConfig(
        num_res_blocks=1,
        num_hidden=1,
        learning_rate=0.001,
        weight_decay=0.0001,
        num_iterations=1,
        n_mcts_sims=2,
        n_games_per_iteration=2,
        n_epochs_per_iteration=1,
        epoch_batch_size=5,
        mask_invalid_actions=False,
        experiment_description=""
    )
    eval_config = EvalConfig(
        n_games=1,
        n_mcts_sims=2,
        opponents=[
                ("random", RandomAgent())
            ]
    )
    device = "cpu"
    # device = "cuda"

    net = ResNet(
        num_res_blocks=train_config.num_res_blocks, num_hidden=train_config.num_hidden, device=device
    )
    optimiser = Adam(net.model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)

    try:
        while True:
            train(
                net,
                optimiser,
                train_config,
                device=device,
            )
            results = list(eval_net(net, eval_config, device))
            print_eval_metrics(results)
    except KeyboardInterrupt:
        pass
        # print("saving & stopping...")
        # az.save(model, num_res_blocks, num_hidden, saved_model_path)

@dataclass
class TrainingMetrics:
    games_played: list[int] = field(default_factory=list)
    policy_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)


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
        print(f"train: {train_config.n_games_per_iteration / dur} games/sec")
        iter_policy_losses.append(pl)
        iter_value_losses.append(vl)

        train_metrics.games_played.append((iteration + 1) * train_config.n_games_per_iteration)
        avg_pl = sum(iter_policy_losses) / len(iter_policy_losses)
        avg_vl = sum(iter_value_losses) / len(iter_value_losses)
        train_metrics.policy_losses.append(avg_pl)
        train_metrics.value_losses.append(avg_vl)


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

def eval_net(net: AzNet, config: EvalConfig, device):
    aza = make_az_agent(net, config.n_mcts_sims, device)
    for oname, oagent in config.opponents:
        w,ll,d = play_games_parallel(aza, oagent, config.n_games)
        yield MatchResults(p1_name="az", p2_name=oname, p1_wins=w, p1_losses=ll, draws=d)


def print_eval_metrics(results: list[MatchResults]):
    for r in results:
        print(f"{r.p1_name} vs {r.p2_name}. p1 WLD: {r.p1_win_rate:.2f} {r.p1_loss_rate:.2f} {r.draw_rate:.2f}")


# def plot_training_metrics(
#     games_played,
#     policy_losses,
#     value_losses,
#     win_rates,
#     loss_rates,
#     win_rates_m10rr,
#     loss_rates_m10rr,
#     win_rates_perfect,
#     loss_rates_perfect,
#     config,
#     custom_text="bleep bloop",
# ):
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#
#     axes[0, 0].plot(games_played, policy_losses, marker="o")
#     axes[0, 0].set_xlabel("Games")
#     axes[0, 0].set_ylabel("Policy Loss")
#     axes[0, 0].set_title("Policy Loss")
#     axes[0, 0].grid(True)
#
#     axes[0, 1].plot(games_played, value_losses, marker="o")
#     axes[0, 1].set_xlabel("Games")
#     axes[0, 1].set_ylabel("Value Loss")
#     axes[0, 1].set_title("Value Loss")
#     axes[0, 1].grid(True)
#
#     axes[1, 0].plot(games_played, win_rates, marker="o", label="vs Random")
#     axes[1, 0].plot(games_played, win_rates_m10rr, marker="s", label="vs m10rr")
#     axes[1, 0].plot(games_played, win_rates_perfect, marker="^", label="vs Perfect")
#     axes[1, 0].set_xlabel("Games")
#     axes[1, 0].set_ylabel("Win Rate")
#     axes[1, 0].set_title("Win Rates")
#     axes[1, 0].set_ylim([0, 1])
#     axes[1, 0].legend()
#     axes[1, 0].grid(True)
#
#     axes[1, 1].plot(games_played, loss_rates, marker="o", label="vs Random")
#     axes[1, 1].plot(games_played, loss_rates_m10rr, marker="s", label="vs m10rr")
#     axes[1, 1].plot(games_played, loss_rates_perfect, marker="^", label="vs Perfect")
#     axes[1, 1].set_xlabel("Games")
#     axes[1, 1].set_ylabel("Loss Rate")
#     axes[1, 1].set_title("Loss Rates")
#     axes[1, 1].set_ylim([0, 1])
#     axes[1, 1].legend()
#     axes[1, 1].grid(True)
#
#     config_text = "Configuration:\n"
#     config_text += f"Res Blocks: {config['num_res_blocks']}\n"
#     config_text += f"Hidden Units: {config['num_hidden']}\n"
#     config_text += f"Batch Size: {config['batch_size']}\n"
#     config_text += f"MCTS Sims: {config['n_mcts_sims']}\n"
#     config_text += f"Games/Update: {config['n_games_per_update']}\n"
#     config_text += f"Epochs/Update: {config['n_epochs_per_update']}\n"
#     config_text += f"LR: {config['learning_rate']}\n"
#     config_text += f"Weight Decay: {config['weight_decay']}"
#
#     if custom_text:
#         config_text += f"\n\n{custom_text}"
#
#     fig.text(
#         0.02,
#         0.02,
#         config_text,
#         fontsize=8,
#         family="monospace",
#         verticalalignment="bottom",
#         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
#     )
#
#     plt.tight_layout(rect=(0, 0.08, 1, 1))  # Leave space for text box
#     plt.savefig(PROJECT_ROOT / "experiment_logs/train_az.png", dpi=150)
#     plt.show()


if __name__ == "__main__":
    main()
