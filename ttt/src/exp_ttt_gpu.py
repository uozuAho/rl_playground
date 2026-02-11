"""Experiment trying to get 100% GPU usage.

What I found
- multiprocessing is easy
- torch tensors are inefficient to send over queues
- np arrays are efficient to send over queues
- pre-pack batches on the producer side
- max game steps per second peaks around a batch size of 1024
"""

import multiprocessing as mp
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import queue as Q

from agents.alphazero import GameStep
from agents.az_nets import ResNet

import ttt.env as t3


def gen_game_step():
    board = np.random.sample(9).tolist()
    player = random.choice([-1, 1])
    mask = [True] * 9
    probs = np.random.sample(9)
    probs /= np.sum(probs)
    val = random.choice([-1, 0, 1])
    return GameStep(board, player, mask, probs.tolist(), val)


def producer_process(queue: mp.Queue, nsteps: int, stop_event: mp.Event):
    while not stop_event.is_set():
        steps = [gen_game_step() for _ in range(nsteps)]
        raw_batch = steps_to_raw_batch(steps)
        try:
            queue.put(raw_batch, timeout=0.1)
        except Q.Full:
            pass


def consumer_process(queue: mp.Queue, nsteps: int, stop_event: mp.Event, profile: bool):
    device = "cuda"
    net = ResNet(4, 64, device)
    optimiser = Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

    last_print_time = time.perf_counter()
    total_update_time = 0.0
    update_count = 0
    start = time.perf_counter()

    while not stop_event.is_set():
        try:
            raw_batch = queue.get(timeout=0.1)
        except Q.Empty:
            continue

        update_start = time.perf_counter()
        states, ptargets, vtargets = raw_batch_to_tensors(raw_batch, device)
        update_net(net, optimiser, states, ptargets, vtargets)
        update_time = time.perf_counter() - update_start
        total_update_time += update_time
        update_count += 1

        # Print metrics once per second
        now = time.perf_counter()
        if now - last_print_time >= 1.0:
            elapsed = now - last_print_time
            avg_update_time = (
                total_update_time / update_count if update_count > 0 else 0
            )
            updates_per_sec = update_count / elapsed

            print(
                f"avg update time: {avg_update_time * 1000:.2f}ms | "
                + f"updates/sec: {updates_per_sec:.2f} | "
                + f"steps/sec: {updates_per_sec * nsteps:.0f} | "
                + f"queue size: {queue.qsize()}"
            )

            last_print_time = now
            total_update_time = 0.0
            update_count = 0

        if profile and time.perf_counter() - start > 3:
            stop_event.set()
            break


def run(profile=False):
    nsteps = 2048
    queue = mp.Queue(maxsize=10)  # Limit queue size to avoid unbounded memory
    stop_event = mp.Event()

    producer = mp.Process(target=producer_process, args=(queue, nsteps, stop_event))
    consumer = mp.Process(
        target=consumer_process, args=(queue, nsteps, stop_event, profile)
    )

    producer.start()
    consumer.start()

    try:
        consumer.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    finally:
        producer.join(timeout=1)
        consumer.join(timeout=1)
        if producer.is_alive():
            producer.terminate()
        if consumer.is_alive():
            consumer.terminate()


def steps_to_raw_batch(game_steps: list[GameStep]):
    """Convert game steps to raw numpy arrays for efficient queue transfer & batch unpacking"""
    batch_size = len(game_steps)

    boards_np = np.empty((batch_size, 3, 3, 3), dtype=np.float32)
    policy_np = np.empty((batch_size, 9), dtype=np.float32)
    value_np = np.empty((batch_size, 1), dtype=np.float32)

    for i, g in enumerate(game_steps):
        boards_np[i, 0] = np.array(
            [c == g.player for c in g.board], dtype=np.float32
        ).reshape(3, 3)
        boards_np[i, 1] = np.array(
            [c == t3.EMPTY for c in g.board], dtype=np.float32
        ).reshape(3, 3)
        boards_np[i, 2] = np.array(
            [c == t3.other_player(g.player) for c in g.board], dtype=np.float32
        ).reshape(3, 3)
        policy_np[i] = g.mcts_probs
        value_np[i, 0] = g.final_value

    return boards_np, policy_np, value_np


def raw_batch_to_tensors(raw_batch, device):
    """Convert raw numpy batch directly to GPU tensors."""
    boards_np, policy_np, value_np = raw_batch
    states = torch.from_numpy(boards_np).to(device)
    policy_targets = torch.from_numpy(policy_np).to(device)
    value_targets = torch.from_numpy(value_np).to(device)
    return states, policy_targets, value_targets


def update_net(
    model: nn.Module,
    optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
):
    out_policy, out_value = model(states)

    policy_loss = F.cross_entropy(out_policy, policy_targets)
    value_loss = F.mse_loss(out_value, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run(profile="profile" in sys.argv)
