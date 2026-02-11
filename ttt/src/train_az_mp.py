"""Same goal as train az, but use alphazero mp (multiprocessing), aiming for\
max GPU usage and therefore fastest possible training"""
from pathlib import Path

import agents.alphazero_mp as az


PROJ_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJ_ROOT/"train_az_mp.log"


def main():
    config = az.Config(
        num_res_blocks=4,
        num_hidden=64,
        learning_rate=0.001,
        weight_decay=0.0001,
        mask_invalid_actions=True,
        train_n_mcts_sims=5,
        c_puct=2.0,
        temperature=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        n_player_processes=1,
        player_n_parallel_games=40,
        batch_size=512,
        weights_update_interval=10,
        device_player="cuda",
        device_learn="cuda",
        stop_after_n_seconds=None,
        stop_after_n_learns=None,
        log_to_file=True,
        log_format_file="text",
        console_log_level="DEBUG",
        log_file_path=LOG_PATH,
    )
    az.train_mp(config)


if __name__ == "__main__":
    main()
