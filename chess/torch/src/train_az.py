import os
from pathlib import Path

import agents.alphazero.azmp as az
from agents.random import RandomAgent

PROJ_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJ_ROOT / "train_az_mp.log"


def main():
    config = az.Config(
        num_res_blocks=1,
        num_hidden=2,
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
        device_player="cuda",
        device_learn="cuda",
        device_eval="cpu",
        cli_log_mode="eval",
        log_to_file=True,
        log_format_file="json",
        console_log_level="INFO",
        log_file_path=LOG_PATH,
    )
    az.train_mp(config)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    az.train_mp(config)


if __name__ == "__main__":
    main()
