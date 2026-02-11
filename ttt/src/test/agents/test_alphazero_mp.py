import agents.alphazero_mp as az


def test_az_mp():
    config = az.Config(
        num_res_blocks=1,
        num_hidden=1,
        learning_rate=0.001,
        weight_decay=0.0001,
        mask_invalid_actions=True,
        train_n_mcts_sims=2,
        c_puct=2.0,
        temperature=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        n_player_processes=1,
        player_n_parallel_games=2,
        batch_size=16,
        weights_update_interval=10,
        device_player="cpu",
        device_learn="cpu",
        stop_after_n_seconds=None,
        stop_after_n_learns=1,
    )
    az.train_mp(config)
