import multiprocessing as mp

import agents.alphazero.azmp as az
from agents.alphazero.azmp import player_loop
from agents.random import RandomAgent


def main():
    run_player(mp.Queue())


def run_player(metrics_queue: mp.Queue):
    config = az.Config(
        num_res_blocks=2,
        num_hidden=48,
        learning_rate=0.001,
        weight_decay=0.0001,
        mask_invalid_actions=True,
        train_n_mcts_sims=60,
        train_halfmove_limit=10,
        train_c_puct=2.0,
        temperature=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.15,
        n_player_processes=1,
        player_n_parallel_games=4,
        epoch_size=512,
        n_epoch_repeats=4,
        batch_size=128,
        weights_update_interval=1,
        discard_on_weight_update=False,
        eval_opponents=[
            ("rng", RandomAgent()),
        ],
        eval_n_games=10,
        eval_n_mcts_sims=10,
        device_player="cuda",
        device_learn="cuda",
        device_eval="cuda",
        cli_log_mode="perf",
        log_to_file=False,
        log_format_file="json",
        console_log_level="INFO",
    )
    player_loop("player", mp.Queue(), mp.Queue(), metrics_queue, mp.Event(), config)


if __name__ == "__main__":
    main()
