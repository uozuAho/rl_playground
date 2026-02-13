"""AlphaZero training with multiprocessing for max GPU utilisation."""

import json
import logging
import multiprocessing as mp
import queue
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import TypedDict

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

import agents.agent
from agents.alphazero import (
    GameStep,
    _self_play_n_games,
    _batch_eval_for_mcts,
    make_az_agent,
)
from agents.az_nets import ResNet
from utils.play import play_games_parallel


@dataclass
class Config:
    num_res_blocks: int = 1
    num_hidden: int = 1

    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    mask_invalid_actions: bool = True

    train_n_mcts_sims: int = 5
    train_c_puct: float = 2.0
    temperature: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    n_player_processes: int = 1
    player_n_parallel_games: int = 8
    epoch_size: int = 1024  # num game steps sent for training
    n_epoch_repeats: int = 4  # num times each epoch is trained on
    batch_size: int = 128  # num game steps trained
    weights_update_interval: int = 10

    eval_c_puct: float = 1.0
    eval_n_mcts_sims: int = 10
    eval_opponents: list[tuple[str, agents.agent.Agent]] = None
    eval_n_games: int = 20

    device_player: str = "cuda"
    device_learn: str = "cuda"
    device_eval: str = "cuda"
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
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if config.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.console_log_level.upper()))
        console_handler.setFormatter(
            text_formatter if config.log_format_console == "text" else json_formatter
        )
        logger.addHandler(console_handler)

    if config.log_to_file:
        if config.log_file_path is None:
            raise ValueError("log_file_path must be set when log_to_file is True")

        log_path = Path(config.log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, config.file_log_level.upper()))
        file_handler.setFormatter(
            text_formatter if config.log_format_file == "text" else json_formatter
        )
        logger.addHandler(file_handler)

    return logger


class PlayerMetrics(TypedDict):
    type: str
    name: str
    games_played: int
    games_per_sec: float
    steps_per_sec: float


def clear_queue(q: mp.Queue):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def player_loop(
    name: str,
    step_queue: mp.Queue,
    weights_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    # logger = setup_logging(config, name)
    net = ResNet(config.num_res_blocks, config.num_hidden, config.device_player)
    net.eval()

    def batch_mcts_eval(envs):
        return _batch_eval_for_mcts(net, envs, config.device_player)

    t_start = time.perf_counter()
    steps_generated = 0
    games_played = 0

    try:
        while not stop_event.is_set():
            try:
                new_state_dict = weights_queue.get_nowait()
                net.model.load_state_dict(new_state_dict)
                clear_queue(step_queue)
            except queue.Empty:
                pass

            with torch.no_grad():
                game_steps = list(
                    _self_play_n_games(
                        batch_mcts_eval,
                        config.player_n_parallel_games,
                        config.train_n_mcts_sims,
                        config.train_c_puct,
                        config.temperature,
                        config.dirichlet_alpha,
                        config.dirichlet_epsilon,
                    )
                )

            games_played += config.player_n_parallel_games
            steps_generated += len(game_steps)
            elapsed = time.perf_counter() - t_start
            metrics_queue.put(
                PlayerMetrics(
                    type="player",
                    name=name,
                    games_played=games_played,
                    games_per_sec=games_played / elapsed,
                    steps_per_sec=steps_generated / elapsed,
                )
            )

            try:
                step_queue.put(game_steps, timeout=1.0)
            except queue.Full:
                pass
    except KeyboardInterrupt:
        pass


class EvalMetrics(TypedDict):
    type: str
    win_rates: dict[str, float]
    loss_rates: dict[str, float]
    draw_rates: dict[str, float]


def eval_loop(
    weights_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    net = ResNet(config.num_res_blocks, config.num_hidden, config.device_eval)
    net.eval()
    aza = make_az_agent(
        net, config.eval_n_mcts_sims, config.train_c_puct, config.device_eval
    )

    def play_eval_games():
        eval_metrics = EvalMetrics(
            type="eval", win_rates={}, loss_rates={}, draw_rates={}
        )

        with torch.no_grad():
            for oname, oagent in config.eval_opponents:
                for azplayer in ["x", "o"]:
                    players = (aza, oagent) if azplayer == "x" else (oagent, aza)
                    pnames = ("az", oname) if azplayer == "x" else (oname, "az")
                    w, ll, d = play_games_parallel(
                        players[0], players[1], config.eval_n_games
                    )
                    matchup = f"{pnames[0]} vs {pnames[1]}"
                    eval_metrics["win_rates"][matchup] = w / config.eval_n_games
                    eval_metrics["loss_rates"][matchup] = ll / config.eval_n_games
                    eval_metrics["draw_rates"][matchup] = d / config.eval_n_games

        return eval_metrics

    try:
        while not stop_event.is_set():
            try:
                new_state_dict = weights_queue.get(timeout=0.1)
                net.model.load_state_dict(new_state_dict)
                metrics = play_eval_games()
                metrics_queue.put(metrics)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass


class BatcherMetrics(TypedDict):
    type: str
    step_queue_size: int
    batch_queue_size: int


def batcher_loop(
    step_queue: mp.Queue,
    epoch_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    buffer = deque()

    last_time = time.perf_counter()
    while not stop_event.is_set():
        if time.perf_counter() - last_time > 1.0:
            last_time = time.perf_counter()
            metrics_queue.put(
                BatcherMetrics(
                    type="batcher",
                    step_queue_size=step_queue.qsize(),
                    batch_queue_size=epoch_queue.qsize(),
                )
            )

        try:
            steps = step_queue.get(timeout=0.1)
            buffer.extend(steps)

            while len(buffer) >= config.epoch_size:
                epoch_steps = []
                for _ in range(config.epoch_size):
                    epoch_steps.append(buffer.popleft())
                random.shuffle(epoch_steps)
                raw_epoch = steps_to_raw_epoch(epoch_steps)
                epoch_queue.put(raw_epoch, timeout=0.1)

        except queue.Empty:
            pass
        except queue.Full:
            pass
        except KeyboardInterrupt:
            break


class LearnerMetrics(TypedDict):
    type: str
    policy_loss: float
    value_loss: float
    steps_trained: int
    steps_per_sec: float


def learner_loop(
    epoch_queue: mp.Queue,
    weights_queues: list[mp.Queue],
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    logger = setup_logging(config, "learner")
    net = ResNet(config.num_res_blocks, config.num_hidden, config.device_learn)
    net.train()
    optimizer = Adam(
        net.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    step_count = 0
    epoch_count = 0
    t_start = time.perf_counter()

    def send_weights():
        state_dict = net.model.state_dict()
        for wq in weights_queues:
            try:
                wq.put(state_dict)
            except queue.Full:
                raise

    def do_learn(raw_batch):
        nonlocal step_count, epoch_count
        policy_losses = []
        value_losses = []

        states, policy_targets, value_targets, valid_action_masks = (
            raw_epoch_to_tensors(raw_batch, config.device_learn)
        )

        for _ in range(config.n_epoch_repeats):
            for i in range(0, config.epoch_size, config.batch_size):
                policy_loss, value_loss = update_net(
                    net,
                    optimizer,
                    states[i : i + config.batch_size],
                    policy_targets[i : i + config.batch_size],
                    value_targets[i : i + config.batch_size],
                    valid_action_masks[i : i + config.batch_size],
                    config.mask_invalid_actions,
                )
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

        step_count += len(states)
        epoch_count += 1

        # Periodically share weights with player processes
        if epoch_count % config.weights_update_interval == 0:
            send_weights()
            clear_queue(epoch_queue)

        elapsed = time.perf_counter() - t_start
        steps_per_sec = step_count / elapsed

        avg_pl = sum(policy_losses) / len(policy_losses) if policy_losses else 0
        avg_vl = sum(value_losses) / len(value_losses) if value_losses else 0

        metrics_queue.put(
            LearnerMetrics(
                type="learner",
                policy_loss=avg_pl,
                value_loss=avg_vl,
                steps_trained=step_count,
                steps_per_sec=steps_per_sec,
            )
        )

    while not stop_event.is_set():
        try:
            if config.stop_after_n_learns and epoch_count >= config.stop_after_n_learns:
                logger.info(f"reached {config.stop_after_n_learns} updates, stopping")
                stop_event.set()
                break
            batch = epoch_queue.get(timeout=0.1)
            do_learn(batch)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break


def metrics_loop(
    metrics_queue: mp.Queue,
    stop_event: mp.Event,
    config: Config,
):
    logger = setup_logging(config, "metrics")
    metric_storage_size = 10
    player_metrics = deque(maxlen=metric_storage_size)
    batcher_metrics = deque(maxlen=metric_storage_size)
    learner_metrics = deque(maxlen=metric_storage_size)
    eval_metrics = deque(maxlen=metric_storage_size)
    metric_metrics = deque(maxlen=metric_storage_size)

    log_time = time.perf_counter()

    while not stop_event.is_set():
        try:
            metrics = metrics_queue.get(timeout=0.1)
            logger.debug(metrics)
            match metrics["type"]:
                case "player":
                    player_metrics.append(metrics)
                case "batcher":
                    batcher_metrics.append(metrics)
                case "learner":
                    learner_metrics.append(metrics)
                case "eval":
                    eval_metrics.append(metrics)
                case "metrics":
                    metric_metrics.append(metrics)
                case _:
                    raise Exception(f"unknown metrics type {metrics['type']}")

            if time.perf_counter() - log_time > 1.0:
                log_time = time.perf_counter()
                metrics_queue.put(
                    {"type": "metrics", "metrics_queue": metrics_queue.qsize()}
                )
                for m in [learner_metrics, eval_metrics]:
                    if m:
                        pprint(m[-1])
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            break


def steps_to_raw_epoch(game_steps: list[GameStep]):
    """Convert game steps to raw numpy arrays for efficient queue transfer."""
    boards = np.stack([ResNet.state2np(x.state) for x in game_steps], dtype=np.float32)
    policy = np.stack([x.mcts_probs for x in game_steps], dtype=np.float32)
    value = np.stack([x.final_value for x in game_steps], dtype=np.float32).reshape(
        (len(game_steps), 1)
    )
    masks = np.stack([x.valid_action_mask for x in game_steps], dtype=np.bool_)

    return boards, policy, value, masks


def raw_epoch_to_tensors(raw_epoch, device):
    """Convert raw numpy batch directly to GPU tensors."""
    boards_np, policy_np, value_np, masks_np = raw_epoch
    states = torch.from_numpy(boards_np).to(device)
    policy_targets = torch.from_numpy(policy_np).to(device)
    value_targets = torch.from_numpy(value_np).to(device)
    valid_action_masks = torch.from_numpy(masks_np).to(device)
    return states, policy_targets, value_targets, valid_action_masks


def update_net(
    net: ResNet,
    optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    valid_action_masks: torch.Tensor,
    mask_invalid_actions: bool,
):
    """Update the network on a batch. Returns policy_loss, value_loss."""
    out_policy, out_value = net.forward_states_tensor(states)

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

    # queue item = list of steps
    step_queue = mp.Queue(maxsize=config.n_player_processes)
    epoch_queue = mp.Queue(maxsize=2)
    metrics_queue = mp.Queue(maxsize=1000)
    weights_queues = [
        mp.Queue(maxsize=1) for _ in range(config.n_player_processes + 1)
    ]  # +1 for evaluator
    stop_event = mp.Event()

    processes = []

    metrics_logger = mp.Process(
        target=metrics_loop,
        name="metrics",
        args=(metrics_queue, stop_event, config),
    )
    processes.append(metrics_logger)

    for i in range(config.n_player_processes):
        name = f"player-{i}"
        p = mp.Process(
            target=player_loop,
            name=name,
            args=(
                name,
                step_queue,
                weights_queues[i],
                metrics_queue,
                stop_event,
                config,
            ),
        )
        processes.append(p)

    p = mp.Process(
        target=batcher_loop,
        name="batcher",
        args=(step_queue, epoch_queue, metrics_queue, stop_event, config),
    )
    processes.append(p)

    if config.eval_opponents:
        p = mp.Process(
            target=eval_loop,
            name="evaluator",
            args=(weights_queues[-1], metrics_queue, stop_event, config),
        )
        processes.append(p)

    learner = mp.Process(
        target=learner_loop,
        name="learner",
        args=(
            epoch_queue,
            weights_queues,
            metrics_queue,
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
