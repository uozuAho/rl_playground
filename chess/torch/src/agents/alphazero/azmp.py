"""AlphaZero training with multiprocessing for max GPU utilisation."""

import json
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event
import queue
import random
import sys
import time
import typing
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import TypedDict

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from agents.agent import ChessAgent
from agents.mctsnew import MctsAgent, best_by_visit_value
from algs.pmcts import ParallelMcts, MCTSNode
from env import env
from agents.alphazero.az_nets import ResNet, AzNet
from utils import types, maths
from utils.maths import heat_dict
from utils.play import play_games_parallel
from utils.types import MPV, MoveProbs


@dataclass
class Config:
    num_res_blocks: int = 1
    num_hidden: int = 1

    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    # mask: speeds training by zeroing policy outputs for invalid actions
    mask_invalid_actions: bool = True

    train_n_mcts_sims: int = 5
    train_halfmove_limit: int | None = None
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

    # discard: when NN weights are updated by training, discard any pending training
    #    steps. No noticeable gain in training efficiency, and increases training time
    #    by 3x.
    discard_on_weight_update: bool = False

    eval_c_puct: float = 1.0
    eval_n_mcts_sims: int = 10
    eval_opponents: list[tuple[str, ChessAgent]] = field(default_factory=list)
    eval_n_games: int = 20

    device_player: str = "cuda"
    device_learn: str = "cuda"
    device_eval: str = "cuda"
    stop_after_n_seconds: float | None = None
    stop_after_train_steps: int | None = None

    # Logging
    # cli_log_mode: perf or eval. perf is for tuning for max training throughput.
    #               eval is for assessing agent strength
    cli_log_mode: str = "perf"
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

    if config.cli_log_mode is not None:
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


@dataclass
class GameStep:
    state: env.ChessGame
    legal_move_mcts_probs: MoveProbs
    final_value: float

    def __repr__(self):
        p = "W" if self.state.turn == env.WHITE else "B"
        b = self.state.fen()
        m = ",".join(f"{m}: {p:.2f}" for m, p in self.legal_move_mcts_probs.items())
        return f"{p} {b} {self.final_value:0.2f}  [{m}]"


class PlayerMetrics(TypedDict):
    type: str
    name: str
    games_played: int
    games_per_sec: float
    steps_per_sec: float
    steps_generated: int
    steps_discarded: int
    avg_steps_per_game: float
    utilisation: float


def clear_queue(q: mp.Queue, count_fn=None):
    """returns number of queue elements cleared"""
    n = 0
    try:
        while True:
            item = q.get_nowait()
            n += count_fn(item) if count_fn else 1
    except queue.Empty:
        pass
    return n


def player_loop(
    name: str,
    step_queue: mp.Queue,
    weights_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: Event,
    config: Config,
):
    net = ResNet(config.num_res_blocks, config.num_hidden, config.device_player)
    net.eval()

    t_start = time.perf_counter()
    t_work_time = 0.0
    steps_generated = 0
    steps_discarded = 0
    games_played = 0
    game_steps = []

    def batch_mcts_eval(envs) -> list[MPV]:
        return _batch_eval_for_mcts(net, envs, config.device_player)

    def try_update_weights():
        nonlocal steps_discarded
        try:
            new_state_dict = weights_queue.get_nowait()
            net.model.load_state_dict(new_state_dict)
            if config.discard_on_weight_update:
                steps_discarded += clear_queue(step_queue, lambda x: len(x)) + len(
                    game_steps
                )
                game_steps.clear()
        except queue.Empty:
            pass

    def play_games():
        return _self_play_n_games(
            batch_mcts_eval,
            config.player_n_parallel_games,
            config.train_n_mcts_sims,
            config.train_c_puct,
            config.temperature,
            config.dirichlet_alpha,
            config.dirichlet_epsilon,
            config.train_halfmove_limit,
        )

    def send_metrics():
        uptime = time.perf_counter() - t_start
        metrics = PlayerMetrics(
            type="player",
            name=name,
            games_played=games_played,
            games_per_sec=games_played / uptime,
            steps_per_sec=steps_generated / uptime,
            steps_generated=steps_generated,
            steps_discarded=steps_discarded,
            avg_steps_per_game=steps_generated / (games_played if games_played else 1),
            utilisation=t_work_time / uptime,
        )
        metrics_queue.put(metrics)

    try:
        while not stop_event.is_set():
            try_update_weights()
            send_metrics()

            if len(game_steps) > 0:
                try:
                    step_queue.put(game_steps, timeout=0.1)
                    game_steps = []
                except queue.Full:
                    pass
            else:
                t_work_start = time.perf_counter()
                with torch.no_grad():
                    temp_game_steps = list(play_games())
                game_steps.extend(temp_game_steps)

                games_played += config.player_n_parallel_games
                steps_generated += len(temp_game_steps)
                t_work_time += time.perf_counter() - t_work_start

    except KeyboardInterrupt:
        pass


class EvalMetrics(TypedDict):
    type: str
    win_rates: dict[str, float]
    loss_rates: dict[str, float]
    draw_rates: dict[str, float]
    utilisation: float


def eval_loop(
    weights_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: Event,
    config: Config,
):
    net = ResNet(config.num_res_blocks, config.num_hidden, config.device_eval)
    net.eval()
    aza = make_az_agent(
        net, config.eval_n_mcts_sims, config.train_c_puct, config.device_eval
    )
    t_start = time.perf_counter()
    t_work_time = 0.0

    def play_eval_games():
        eval_metrics = EvalMetrics(
            type="eval", win_rates={}, loss_rates={}, draw_rates={}, utilisation=0.0
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
                t_work_start = time.perf_counter()
                net.model.load_state_dict(new_state_dict)
                metrics = play_eval_games()
                t_work_time += time.perf_counter() - t_work_start
                uptime = time.perf_counter() - t_start
                metrics["utilisation"] = t_work_time / uptime
                metrics_queue.put(metrics)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        pass


class BatcherMetrics(TypedDict):
    type: str
    utilisation: float
    buffer_size: int
    step_queue_size: int
    batch_queue_size: int


def batcher_loop(
    step_queue: mp.Queue,
    epoch_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: Event,
    config: Config,
):
    net = ResNet(1, 1, "cpu")
    buffer = deque()
    t_start = time.perf_counter()
    t_work = 0.0
    emit_metrics_time = time.perf_counter()
    next_epoch = None

    def emit_metrics():
        uptime = time.perf_counter() - t_start
        utilisation = t_work / uptime
        metrics_queue.put(
            BatcherMetrics(
                type="batcher",
                buffer_size=len(buffer),
                step_queue_size=step_queue.qsize(),
                batch_queue_size=epoch_queue.qsize(),
                utilisation=utilisation,
            )
        )

    while not stop_event.is_set():
        if time.perf_counter() - emit_metrics_time > 1.0:
            emit_metrics_time = time.perf_counter()
            emit_metrics()

        try:
            while len(buffer) < config.epoch_size:
                steps = step_queue.get(timeout=0.1)
                buffer.extend(steps)
            if next_epoch is not None:
                epoch_queue.put(next_epoch, timeout=0.1)
                next_epoch = None
            else:
                t_start_work = time.perf_counter()
                epoch_steps = []
                for i in range(config.epoch_size):
                    epoch_steps.append(buffer.popleft())
                random.shuffle(epoch_steps)
                next_epoch = steps_to_raw_epoch(net, epoch_steps)
                t_work += time.perf_counter() - t_start_work
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
    epoch_count: int
    epochs_discarded: int
    utilisation: float


def get_state_dict_cpu(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}


def learner_loop(
    epoch_queue: mp.Queue,
    player_weights_queues: list[mp.Queue],
    eval_weight_queue: mp.Queue,
    metrics_queue: mp.Queue,
    stop_event: Event,
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
    epochs_discarded = 0
    t_start = time.perf_counter()
    t_learn_time = 0.0

    def send_weights():
        state_dict = get_state_dict_cpu(net.model)
        for wq in player_weights_queues:
            try:
                wq.put(state_dict)
            except queue.Full:
                raise Exception("Player weight queue full")
        clear_queue(eval_weight_queue)
        eval_weight_queue.put(state_dict)

    def do_learn(raw_batch):
        nonlocal step_count, epoch_count, epochs_discarded, t_learn_time
        t_start_learn = time.perf_counter()
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
            if config.discard_on_weight_update:
                epochs_discarded += clear_queue(epoch_queue)

        avg_pl = sum(policy_losses) / len(policy_losses) if policy_losses else 0
        avg_vl = sum(value_losses) / len(value_losses) if value_losses else 0

        uptime = time.perf_counter() - t_start
        steps_per_sec = step_count / uptime
        t_learn_time += time.perf_counter() - t_start_learn

        return LearnerMetrics(
            type="learner",
            policy_loss=avg_pl,
            value_loss=avg_vl,
            steps_trained=step_count,
            steps_per_sec=steps_per_sec,
            epoch_count=epoch_count,
            epochs_discarded=epochs_discarded,
            utilisation=t_learn_time / uptime,
        )

    while not stop_event.is_set():
        try:
            if (
                config.stop_after_train_steps
                and step_count >= config.stop_after_train_steps
            ):
                logger.info(
                    f"reached {config.stop_after_train_steps} train steps, stopping"
                )
                stop_event.set()
                break
            batch = epoch_queue.get(timeout=0.1)
            metrics = do_learn(batch)
            metrics_queue.put(metrics)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break


def metrics_loop(
    metrics_queue: mp.Queue,
    stop_event: Event,
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
                match config.cli_log_mode:
                    case "eval":
                        for m in [learner_metrics, eval_metrics]:
                            if m:
                                pprint(m[-1])
                    case "perf":
                        print("-----")
                        for m in [
                            player_metrics,
                            batcher_metrics,
                            learner_metrics,
                            eval_metrics,
                        ]:
                            if m:
                                pprint(m[-1])
                    case _:
                        print(f"unknown cli_log_mode {config.cli_log_mode}")
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            break


def steps_to_raw_epoch(net: AzNet, game_steps: list[GameStep]):
    """Convert game steps to raw numpy arrays for efficient queue transfer."""
    codec = net.get_codec()
    boards = np.stack([net.state2np(x.state) for x in game_steps], dtype=np.float32)
    policy = np.stack(
        [codec.dict2prior(x.legal_move_mcts_probs) for x in game_steps],
        dtype=np.float32,
    )
    value = np.stack([x.final_value for x in game_steps], dtype=np.float32).reshape(
        (len(game_steps), 1)
    )
    masks = np.stack([codec.validmask(x.state) for x in game_steps], dtype=np.bool_)

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
    epoch_queue = mp.Queue(maxsize=1)
    metrics_queue = mp.Queue(maxsize=1000)
    player_weights_queues = [
        mp.Queue(maxsize=1) for _ in range(config.n_player_processes)
    ]
    eval_weight_queue = mp.Queue(maxsize=1)
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
                player_weights_queues[i],
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
            args=(eval_weight_queue, metrics_queue, stop_event, config),
        )
        processes.append(p)

    learner = mp.Process(
        target=learner_loop,
        name="learner",
        args=(
            epoch_queue,
            player_weights_queues,
            eval_weight_queue,
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
            logger.info(f"Reached {config.stop_after_n_seconds} seconds, stopping...")
            stop_event.set()
        while True:
            if any(p.exitcode not in [None, 0] for p in processes):
                logger.error("Something died, stopping all processes...")
                stop_event.set()
                break
            learner.join(timeout=1)
            if not learner.is_alive():
                break
    except KeyboardInterrupt:
        logger.info("User requested stop. Stopping...")
        stop_event.set()
    finally:
        for p in processes:
            logger.info(f"Waiting for {p}")
            p.join(timeout=1)
            if p.is_alive():
                p.terminate()
            if p.exitcode not in [None, 0]:
                raise ChildProcessError("Something died")


def _batch_eval_for_mcts(net: AzNet, states: list[env.ChessGame], device) -> list[MPV]:
    return net.mpv_batch(states)


def _mcts_probs(root: MCTSNode) -> MoveProbs:
    total_visits = sum(c.visits for c in root.children.values())
    assert total_visits > 0
    probs = {move: node.visits / total_visits for move, node in root.children.items()}
    assert maths.is_prob_dist(probs.values())
    return probs


def _self_play_n_games(
    eval_fn: types.BatchEvaluateFunc,
    n_games: int,
    n_mcts_sims: int,
    c_puct: float,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    halfmove_limit: int | None,
) -> typing.Iterable[GameStep]:
    states = [env.ChessGame(halfmove_limit=halfmove_limit) for _ in range(n_games)]
    game_overs = [False for _ in range(n_games)]
    trajectories = [[] for _ in range(n_games)]
    winners: list[bool | None | env.Player] = [False for _ in range(n_games)]
    while not all(game_overs):
        active_idxs = [i for i, go in enumerate(game_overs) if not go]
        active_envs = [states[i] for i in active_idxs]
        roots = ParallelMcts(
            active_envs,
            eval_fn,
            num_simulations=n_mcts_sims,
            c_puct=c_puct,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        ).run()
        for i, root in zip(active_idxs, roots):
            state = root.state
            move_probs = _mcts_probs(root)
            trajectories[i].append((state, move_probs))
            move_probs = heat_dict(move_probs, temperature)
            mpi = list(move_probs.items())
            move, _ = random.choices(mpi, [m[1] for m in mpi])[0]
            states[i].do(move)
            if states[i].is_game_over():
                game_overs[i] = True
                winners[i] = states[i].winner()
    assert not any(x is False for x in winners)
    for i, t in enumerate(trajectories):
        for state, move_probs in t:
            winner = winners[i]
            final_reward = 0 if winner is None else 1 if state.turn == winner else -1
            yield GameStep(state, move_probs, final_reward)


def make_az_agent(net: AzNet, n_sims: int, c_puct: float, device):
    return MctsAgent(
        batch_eval_fn=nn_2_batch_eval(net, device),
        select_action_fn=best_by_visit_value,
        n_sims=n_sims,
        c_puct=c_puct,
    )


def nn_2_batch_eval(net: AzNet, device):
    def eval_fn(states: list[env.ChessGame]):
        return _batch_eval_for_mcts(net, states, device)

    return eval_fn
