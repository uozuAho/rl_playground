import torch

from agents import az_nets
import env.connect4 as c4
from utils.maths import is_prob_dist


def test_resnet_pv():
    net = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    state = c4.new_game()
    with torch.no_grad():
        p, v = net.pv(state)
    assert len(p) == c4.ACTION_SIZE
    assert is_prob_dist(p)
    assert type(v) is float


def test_resnet_pv_batch():
    net = az_nets.ResNet(num_res_blocks=1, num_hidden=1, device="cpu")
    batch_size = 2
    states = [c4.new_game() for _ in range(batch_size)]
    with torch.no_grad():
        pvs = net.pv_batch(states)
    assert len(pvs) == batch_size
    for p, v in pvs:
        assert len(p) == c4.ACTION_SIZE
        assert is_prob_dist(p)
        assert type(v) is float
