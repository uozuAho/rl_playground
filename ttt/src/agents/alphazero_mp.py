"""AlphaZero training with multiprocessing for GPU optimization.

Architecture:
- Player processes: Run self-play games, put game steps on a queue
- Batching process: Reads game steps, writes numpy array batches to another queue
- Learning process: Reads numpy batches, updates the NN model
"""

import itertools
import multiprocessing as mp
import random
import time
import typing
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

import algs.az_mcts as mcts
from agents.agent import TttAgent
from agents.alphazero import (
    GameStep,
    _self_play_n_games,
    _board2tensor,
    _batch_eval_for_mcts,
    nn_2_batch_eval,
)
from agents.az_nets import ResNet
import ttt.env as t3


def player_process(
    step_queue: mp.Queue,
    weights_queue: mp.Queue,
    stop_event: mp.Event,
    n_games_per_iter: int,
    n_mcts_sims: int,
    c_puct: float,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    device: str,
    num_res_blocks: int,
    num_hidden: int,
):
    """Self-play process that generates game steps.

    Maintains its own copy of the NN for MCTS evaluation.
    Updates weights from the learning process when available.
    """
    # Create local model for MCTS evaluation
    model = ResNet(num_res_blocks, num_hidden, device)
    model.eval()

    def batch_mcts_eval(envs):
        return _batch_eval_for_mcts(model, envs, device)

    while not stop_event.is_set():
        # Check for weight updates from learning process
        try:
            new_state_dict = weights_queue.get_nowait()
            model.load_state_dict(new_state_dict)
        except:
            pass

        # Generate game steps via self-play
        with torch.no_grad():
            game_steps = list(
                _self_play_n_games(
                    batch_mcts_eval,
                    n_games_per_iter,
                    n_mcts_sims,
                    c_puct,
                    temperature,
                    dirichlet_alpha,
                    dirichlet_epsilon,
                )
            )

        # Put game steps on queue for batching process
        for step in game_steps:
            try:
                step_queue.put(step, timeout=0.1)
            except:
                # Queue full, skip this step
                pass


def batching_process(
    step_queue: mp.Queue,
    batch_queue: mp.Queue,
    stop_event: mp.Event,
    batch_size: int,
):
    """Batching process that reads game steps and creates numpy batches."""
    buffer = []

    while not stop_event.is_set():
        try:
            step = step_queue.get(timeout=0.1)
            buffer.append(step)

            if len(buffer) >= batch_size:
                # Convert to numpy batch
                raw_batch = steps_to_raw_batch(buffer)
                batch_queue.put(raw_batch, timeout=0.1)
                buffer = []
        except:
            # Timeout on get or put - continue
            pass

    # Flush remaining steps in buffer
    if buffer:
        raw_batch = steps_to_raw_batch(buffer)
        try:
            batch_queue.put(raw_batch, timeout=1.0)
        except:
            pass


def learning_process(
    batch_queue: mp.Queue,
    weights_queues: list[mp.Queue],
    stop_event: mp.Event,
    device: str,
    num_res_blocks: int,
    num_hidden: int,
    learning_rate: float,
    weight_decay: float,
    mask_invalid_actions: bool,
    weights_update_interval: int,
):
    """Learning process that trains the NN on batches."""
    model = ResNet(num_res_blocks, num_hidden, device)
    model.train()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    batch_count = 0
    last_print_time = time.perf_counter()
    total_update_time = 0.0
    update_count = 0
    policy_losses = []
    value_losses = []

    while not stop_event.is_set():
        try:
            raw_batch = batch_queue.get(timeout=0.1)
        except:
            continue

        update_start = time.perf_counter()

        # Convert numpy batch to tensors
        states, policy_targets, value_targets, valid_action_masks = raw_batch_to_tensors(
            raw_batch, device
        )

        # Update network
        policy_loss, value_loss = update_net(
            model,
            optimizer,
            states,
            policy_targets,
            value_targets,
            valid_action_masks,
            mask_invalid_actions,
        )

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        update_time = time.perf_counter() - update_start
        total_update_time += update_time
        update_count += 1
        batch_count += 1

        # Periodically share weights with player processes
        if batch_count % weights_update_interval == 0:
            state_dict = model.state_dict()
            for wq in weights_queues:
                try:
                    # Non-blocking put - if queue is full, skip
                    wq.put_nowait(state_dict)
                except:
                    pass

        # Print metrics once per second
        now = time.perf_counter()
        if now - last_print_time >= 1.0:
            elapsed = now - last_print_time
            avg_update_time = total_update_time / update_count if update_count > 0 else 0
            updates_per_sec = update_count / elapsed
            batch_size = len(states)
            steps_per_sec = updates_per_sec * batch_size

            avg_pl = sum(policy_losses) / len(policy_losses) if policy_losses else 0
            avg_vl = sum(value_losses) / len(value_losses) if value_losses else 0

            print(
                f"policy_loss: {avg_pl:.4f} | "
                + f"value_loss: {avg_vl:.4f} | "
                + f"updates/sec: {updates_per_sec:.2f} | "
                + f"steps/sec: {steps_per_sec:.0f} | "
                + f"batch queue: {batch_queue.qsize()}"
            )

            last_print_time = now
            total_update_time = 0.0
            update_count = 0
            policy_losses = []
            value_losses = []


def steps_to_raw_batch(game_steps: list[GameStep]):
    """Convert game steps to raw numpy arrays for efficient queue transfer."""
    batch_size = len(game_steps)

    boards_np = np.empty((batch_size, 3, 3, 3), dtype=np.float32)
    policy_np = np.empty((batch_size, 9), dtype=np.float32)
    value_np = np.empty((batch_size, 1), dtype=np.float32)
    masks_np = np.empty((batch_size, 9), dtype=np.bool_)

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
        masks_np[i] = g.valid_action_mask

    return boards_np, policy_np, value_np, masks_np


def raw_batch_to_tensors(raw_batch, device):
    """Convert raw numpy batch directly to GPU tensors."""
    boards_np, policy_np, value_np, masks_np = raw_batch
    states = torch.from_numpy(boards_np).to(device)
    policy_targets = torch.from_numpy(policy_np).to(device)
    value_targets = torch.from_numpy(value_np).to(device)
    valid_action_masks = torch.from_numpy(masks_np).to(device)
    return states, policy_targets, value_targets, valid_action_masks


def update_net(
    model: nn.Module,
    optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    valid_action_masks: torch.Tensor,
    mask_invalid_actions: bool,
):
    """Update the network on a batch. Returns policy_loss, value_loss."""
    out_policy, out_value = model(states)

    if mask_invalid_actions:
        out_policy = out_policy.masked_fill(~valid_action_masks, -1e32)

    policy_loss = F.cross_entropy(out_policy, policy_targets)
    value_loss = F.mse_loss(out_value, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def train_mp(
    num_res_blocks: int = 1,
    num_hidden: int = 1,
    device: str = "cuda",
    n_player_processes: int = 4,
    n_games_per_iter: int = 8,
    n_mcts_sims: int = 5,
    c_puct: float = 2.0,
    temperature: float = 1.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    mask_invalid_actions: bool = True,
    weights_update_interval: int = 10,
    duration_seconds: float = None,
):
    """Train AlphaZero using multiprocessing.

    Args:
        num_res_blocks: Number of residual blocks in the network
        num_hidden: Number of hidden units
        device: Device for learning process ("cuda" or "cpu")
        n_player_processes: Number of parallel self-play processes
        n_games_per_iter: Games to play per iteration in each player process
        n_mcts_sims: Number of MCTS simulations per move
        c_puct: PUCT constant for MCTS exploration
        temperature: Temperature for action selection from MCTS probs
        dirichlet_alpha: Alpha parameter for Dirichlet noise
        dirichlet_epsilon: Epsilon for mixing in Dirichlet noise
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        mask_invalid_actions: Whether to mask invalid actions in policy loss
        weights_update_interval: How often (in batches) to share weights with players
        duration_seconds: How long to train (None = run until interrupted)
    """
    mp.set_start_method('spawn', force=True)

    # Create queues
    step_queue = mp.Queue(maxsize=10000)
    batch_queue = mp.Queue(maxsize=100)
    weights_queues = [mp.Queue(maxsize=1) for _ in range(n_player_processes)]
    stop_event = mp.Event()

    # Create processes
    processes = []

    # Player processes
    for i in range(n_player_processes):
        p = mp.Process(
            target=player_process,
            args=(
                step_queue,
                weights_queues[i],
                stop_event,
                n_games_per_iter,
                n_mcts_sims,
                c_puct,
                temperature,
                dirichlet_alpha,
                dirichlet_epsilon,
                device,
                num_res_blocks,
                num_hidden,
            ),
        )
        processes.append(p)

    # Batching process
    p = mp.Process(
        target=batching_process,
        args=(step_queue, batch_queue, stop_event, batch_size),
    )
    processes.append(p)

    # Learning process
    learner = mp.Process(
        target=learning_process,
        args=(
            batch_queue,
            weights_queues,
            stop_event,
            device,
            num_res_blocks,
            num_hidden,
            learning_rate,
            weight_decay,
            mask_invalid_actions,
            weights_update_interval,
        ),
    )
    processes.append(learner)

    # Start all processes
    for p in processes:
        p.start()

    # Run for specified duration or until interrupted
    try:
        if duration_seconds:
            time.sleep(duration_seconds)
            stop_event.set()
        learner.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    finally:
        # Clean up
        for p in processes:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()


if __name__ == "__main__":
    train_mp(
        num_res_blocks=4,
        num_hidden=64,
        device="cuda",
        n_player_processes=4,
        n_games_per_iter=8,
        n_mcts_sims=5,
        batch_size=512,
        duration_seconds=None,  # Run until Ctrl+C
    )
