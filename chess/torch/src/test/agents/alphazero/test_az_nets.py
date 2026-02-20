import torch

from agents.alphazero import az_nets
from utils.maths import is_prob_dist
from env import env


def test_resnet_pv():
    net = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    state = env.ChessGame()
    with torch.no_grad():
        p, v = net.pv(state)
    assert len(p) == net.get_codec().ACTION_SIZE
    assert is_prob_dist(p)
    assert type(v) is float


def test_resnet_pv_batch():
    net = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    for batch_size in range(1, 3):
        states = [env.ChessGame() for _ in range(batch_size)]
        with torch.no_grad():
            pvs = net.pv_batch(states)
        assert len(pvs) == batch_size
        for p, v in pvs:
            assert len(p) == net.get_codec().ACTION_SIZE
            assert is_prob_dist(p)
            assert type(v) is float
