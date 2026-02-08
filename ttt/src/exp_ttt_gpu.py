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
    nsteps = 512

    try:
        runloop(device, net, nsteps, optimiser, profile)
    except KeyboardInterrupt:
        pass


def runloop(device: str, net: ResNet, nsteps: int, optimiser: Adam, profile: bool):
    last_print_time = time.perf_counter()
    total_gen_time = 0.0
    total_update_time = 0.0
    update_count = 0

    start = time.perf_counter()

    while True:
        gen_start = time.perf_counter()
        steps = [gen_game_step() for _ in range(nsteps)]
        gen_time = time.perf_counter() - gen_start
        total_gen_time += gen_time

        update_start = time.perf_counter()
        _update_net(net, optimiser, steps, mask_invalid_actions=False, device=device)
        update_time = time.perf_counter() - update_start
        total_update_time += update_time
        update_count += 1

        # Print metrics once per second
        now = time.perf_counter()
        if now - last_print_time >= 1.0:
            elapsed = now - last_print_time
            avg_gen_time = total_gen_time / update_count if update_count > 0 else 0
            avg_update_time = (
                total_update_time / update_count if update_count > 0 else 0
            )
            updates_per_sec = update_count / elapsed

            print(
                f"avg gen time: {avg_gen_time * 1000:.2f}ms | "
                + f"avg update time: {avg_update_time * 1000:.2f}ms | "
                + f"updates/sec: {updates_per_sec:.2f} = steps/sec: {updates_per_sec * nsteps}"
            )

            last_print_time = now
            total_gen_time = 0.0
            total_update_time = 0.0
            update_count = 0

        if profile and time.perf_counter() - start > 3:
            break


if __name__ == "__main__":
    run(profile="profile" in sys.argv)
