from torch.optim import Adam

import agents.alphazero as az
from agents import az_nets
from agents.simple import RandomAgent
from utils.play import play_games_parallel


def test_train_and_play():
    net = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    optimiser = Adam(net.model.parameters(), lr=0.001, weight_decay=0.0001)
    az.train(
        net,
        optimiser,
        n_games=1,
        n_epochs=1,
        n_mcts_sims=2,
        device="cpu",
        verbose=False,
        train_batch_size=1,
        mask_invalid_actions=False,
    )
    aza = az.make_az_agent(net, n_sims=1, c_puct=1.0, device="cpu")
    play_games_parallel(aza, RandomAgent(), 1)
