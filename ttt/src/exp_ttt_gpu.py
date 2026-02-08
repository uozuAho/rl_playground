"""Trying to get 100% GPU usage"""

import random
import sys
import time

import numpy as np
from torch.optim import Adam

from agents.alphazero import _update_net, GameStep
from agents.az_nets import ResNet


def gen_game_step():
    board = np.random.sample(9).tolist()
    player = random.choice([-1, 1])
    mask = [True] * 9
    probs = np.random.sample(9)
    probs /= np.sum(probs)
    val = random.choice([-1, 0, 1])
    return GameStep(board, player, mask, probs.tolist(), val)


def run(profile=False):
    device = "cuda"
    net = ResNet(4, 64, device)
    optimiser = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    nsteps = 128
    start = time.perf_counter()
    while True:
        steps = [gen_game_step() for _ in range(nsteps)]
        _update_net(net, optimiser, steps, mask_invalid_actions=False, device=device)
        if profile and time.perf_counter() - start > 3:
            break


if __name__ == "__main__":
    run(profile="profile" in sys.argv)
