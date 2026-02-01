import pytest
from torch.optim import Adam

import agents.alphazero as az
from agents import az_nets
from agents.simple import RandomAgent
from utils.play import play_games_parallel


@pytest.mark.skip("fix lower level stuff")
def test_train_and_play():
    model = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    az.train(
        model,
        optimiser,
        n_games=1,
        n_epochs=1,
        n_mcts_sims=1,
        device="cpu",
        verbose=False,
        train_batch_size=1,
        mask_invalid_actions=False,
    )
    aza = az.make_az_agent(model, n_sims=1, device="cpu")
    play_games_parallel(aza, RandomAgent(), 1)
