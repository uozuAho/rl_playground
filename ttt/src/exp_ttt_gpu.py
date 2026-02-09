"""Trying to get 100% GPU usage"""

import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from agents.alphazero import GameStep
from agents.az_nets import ResNet
from utils.maths import is_prob_dist

import ttt.env as t3


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
        update_net(net, optimiser, steps, mask_invalid_actions=False, device=device)
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


def steps_to_batch(
    game_steps: list[GameStep],
):
    """Returns tensors: states, policy targets, value targets"""
    for s in game_steps:
        assert is_prob_dist(s.mcts_probs)
        for i, v in enumerate(s.valid_action_mask):
            if not v:
                assert s.board[i] != t3.EMPTY
                assert s.mcts_probs[i] == 0
    state = torch.stack([board2tensor(g.board, g.player) for g in game_steps])
    policy_targets = torch.tensor([g.mcts_probs for g in game_steps], dtype=torch.float32)
    value_targets = torch.tensor([g.final_value for g in game_steps], dtype=torch.float32).reshape((-1, 1))
    return state, policy_targets, value_targets


def update_net(
    model: nn.Module,
    optimizer,
    game_steps: list[GameStep],
    mask_invalid_actions,
    device,
):
    state, policy_targets, value_targets = steps_to_batch(game_steps)
    state = state.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)

    out_policy, out_value = model(state)

    if mask_invalid_actions:
        valid_action_masks = torch.stack(
            [
                torch.tensor(s.valid_action_mask, dtype=torch.bool, device=device)
                for s in game_steps
            ]
        )
        out_policy = out_policy.masked_fill(~valid_action_masks, -1e32)

    policy_loss = F.cross_entropy(out_policy, policy_targets)
    value_loss = F.mse_loss(out_value, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def board2tensor(board: t3.Board, current_player: t3.Player):
    np_array = np.array(
        [
            [c == current_player for c in board],
            [c == t3.EMPTY for c in board],
            [c == t3.other_player(current_player) for c in board],
        ],
        dtype=np.float32,
    ).reshape(3, 3, 3)
    return torch.from_numpy(np_array)


if __name__ == "__main__":
    run(profile="profile" in sys.argv)
