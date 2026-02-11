"""AlphaZero training with multiprocessing for max GPU utilisation.

This is overkill for tic tac toe, but serves as a POC for bigger games.
Inspired by https://github.com/google-deepmind/open_spiel/open_spiel/python/algorithms/alpha_zero

Architecture:
- Player processes: Run self-play games, put game steps on a queue
- Batching process: Reads game steps, writes numpy array batches to another queue
- Learning process: Reads numpy batches, updates the NN model
"""

import json
import logging
import multiprocessing as mp
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from agents.alphazero import (
    GameStep,
    _self_play_n_games,
    _batch_eval_for_mcts,
)
from agents.az_nets import ResNet
import ttt.env as t3


@dataclass
class Config:
    num_res_blocks: int = 1
    num_hidden: int = 1

    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    mask_invalid_actions: bool = True

    train_n_mcts_sims: int = 5
    c_puct: float = 2.0
    temperature: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    n_player_processes: int = 4
    player_n_parallel_games: int = 8
    batch_size: int = 512
    weights_update_interval: int = 10

    device_player: str = "cuda"
    device_learn: str = "cuda"
    stop_after_n_seconds: float | None = None
    stop_after_n_learns: int | None = None  # convenient for testing, benchmarks

    # Logging
    log_to_console: bool = True
    log_to_file: bool = False
    log_file_path: str | Path | None = None
    log_format_console: str = "text"  # "text" or "json"
    log_format_file: str = "json"  # text or json
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "process": record.processName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logging(config: Config, process_name: str = "main") -> logging.Logger:
    logger = logging.getLogger(f"alphazero_mp.{process_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    json_formatter = JSONFormatter()
    text_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(processName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.console_log_level.upper()))
        console_handler.setFormatter(text_formatter if config.log_format_console == "text" else json_formatter)
        logger.addHandler(console_handler)

    if config.log_to_file:
        if config.log_file_path is None:
            raise ValueError("log_file_path must be set when log_to_file is True")

        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, config.file_log_level.upper()))
        file_handler.setFormatter(text_formatter if config.log_format_file == "text" else json_formatter)
        logger.addHandler(file_handler)

    return logger


def player_process(
    step_queue: mp.Queue, weights_queue: mp.Queue, stop_event: mp.Event, config: Config
):
    logger = setup_logging(config, "player")
    model = ResNet(config.num_res_blocks, config.num_hidden, config.device_player)
    model.eval()

    def batch_mcts_eval(envs):
        return _batch_eval_for_mcts(model, envs, config.device_player)

    try:
        while not stop_event.is_set():
            try:
                new_state_dict = weights_queue.get_nowait()
                model.load_state_dict(new_state_dict)
                logger.debug("updated model weights")
            except queue.Empty:
                pass

            with torch.no_grad():
                game_steps = list(
                    _self_play_n_games(
                        batch_mcts_eval,
                        config.player_n_parallel_games,
                        config.train_n_mcts_sims,
                        config.c_puct,
                        config.temperature,
                        config.dirichlet_alpha,
                        config.dirichlet_epsilon,
                    )
                )

            for step in game_steps:
                try:
                    step_queue.put(step, timeout=0.2)
                except queue.Full:
                    logger.warning("step queue full")
                    pass
    except KeyboardInterrupt:
        pass


def batching_process(
    step_queue: mp.Queue,
    batch_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    logger = setup_logging(config, "batcher")
    batch_size = config.batch_size
    buffer = []

    while not stop_event.is_set():
        try:
            while len(buffer) < batch_size:
                step = step_queue.get(timeout=0.1)
                buffer.append(step)

            if len(buffer) >= batch_size:
                raw_batch = steps_to_raw_batch(buffer)
                batch_queue.put(raw_batch, timeout=0.1)
                buffer = []
        except queue.Empty:
            pass
        except queue.Full:
            pass
        except KeyboardInterrupt:
            break

    if buffer:
        raw_batch = steps_to_raw_batch(buffer)
        try:
            batch_queue.put(raw_batch, timeout=1.0)
        except queue.Full:
            logger.warning("batch queue full")
            pass
        except KeyboardInterrupt:
            pass


def learning_process(
    batch_queue: mp.Queue,
    weights_queues: list[mp.Queue],
    stop_event: mp.Event,
    config: Config,
):
    logger = setup_logging(config, "learner")
    model = ResNet(config.num_res_blocks, config.num_hidden, config.device_learn)
    model.train()
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    batch_count = 0
    last_log_time = time.perf_counter()
    total_update_time = 0.0
    update_count = 0
    policy_losses = []
    value_losses = []

    while not stop_event.is_set():
        try:
            raw_batch = batch_queue.get(timeout=0.1)
        except queue.Empty:
            logger.debug("batch queue empty")
            continue
        except KeyboardInterrupt:
            break

        if config.stop_after_n_learns and update_count >= config.stop_after_n_learns:
            logger.info(f"reached {config.stop_after_n_learns} updates, stopping")
            stop_event.set()
            break

        update_start = time.perf_counter()

        states, policy_targets, value_targets, valid_action_masks = (
            raw_batch_to_tensors(raw_batch, config.device_learn)
        )

        policy_loss, value_loss = update_net(
            model,
            optimizer,
            states,
            policy_targets,
            value_targets,
            valid_action_masks,
            config.mask_invalid_actions,
        )

        policy_losses.append(policy_loss)
        value_losses.append(value_loss)

        update_time = time.perf_counter() - update_start
        total_update_time += update_time
        update_count += 1
        batch_count += 1

        # Periodically share weights with player processes
        if batch_count % config.weights_update_interval == 0:
            state_dict = model.state_dict()
            for wq in weights_queues:
                try:
                    wq.put(state_dict)
                except queue.Full:
                    raise

        # Log metrics once per second
        now = time.perf_counter()
        if now - last_log_time >= 1.0:
            elapsed = now - last_log_time
            updates_per_sec = update_count / elapsed
            batch_size = len(states)
            steps_per_sec = updates_per_sec * batch_size

            avg_pl = sum(policy_losses) / len(policy_losses) if policy_losses else 0
            avg_vl = sum(value_losses) / len(value_losses) if value_losses else 0

            logger.info(
                f"policy_loss: {avg_pl:.4f} | "
                + f"value_loss: {avg_vl:.4f} | "
                + f"updates/sec: {updates_per_sec:.2f} | "
                + f"steps/sec: {steps_per_sec:.0f} | "
                + f"batch queue: {batch_queue.qsize()}"
            )

            last_log_time = now
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


def train_mp(config: Config):
    logger = setup_logging(config, "main")
    mp.set_start_method("spawn", force=True)

    step_queue = mp.Queue(maxsize=1009)
    batch_queue = mp.Queue(maxsize=100)
    weights_queues = [mp.Queue(maxsize=1) for _ in range(config.n_player_processes)]
    stop_event = mp.Event()

    processes = []

    for i in range(config.n_player_processes):
        p = mp.Process(
            target=player_process,
            name=f"player-{i}",
            args=(
                step_queue,
                weights_queues[i],
                stop_event,
                config,
            ),
        )
        processes.append(p)

    p = mp.Process(
        target=batching_process,
        name="batcher",
        args=(step_queue, batch_queue, stop_event, config),
    )
    processes.append(p)

    learner = mp.Process(
        target=learning_process,
        name="learner",
        args=(
            batch_queue,
            weights_queues,
            stop_event,
            config,
        ),
    )
    processes.append(learner)

    for p in processes:
        p.start()

    try:
        if config.stop_after_n_seconds:
            time.sleep(config.stop_after_n_seconds)
            stop_event.set()
        learner.join()
    except KeyboardInterrupt:
        logger.info("Stopping...")
        stop_event.set()
    finally:
        # Clean up
        for p in processes:
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
