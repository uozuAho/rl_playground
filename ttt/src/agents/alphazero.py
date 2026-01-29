import itertools
import random
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import algs.az_mcts as mcts

from agents.agent import TttAgent
import ttt.env as t3
from agents.az_nets import ResNet
from utils.maths import softmax, is_prob_dist

type MctsProbs = list[float]


@dataclass
class GameStep:
    board: t3.Board
    player: t3.Player
    valid_action_mask: list[bool]
    mcts_probs: list[float]
    final_value: float

    def as_tuple(self):
        return (
            self.board,
            self.player,
            self.valid_action_mask,
            self.mcts_probs,
            self.final_value,
        )

    def __repr__(self):
        b = t3.board2str1d(self.board, sep="|")
        p = "X" if self.player == t3.X else "O"
        m = ",".join(f"{x:.2f}" for x in self.mcts_probs)
        # TODO: valid action masks
        return f"{b} {p} {self.final_value:0.2f}  [{m}]"


def load_net(path: Path, device):
    d = torch.load(path, weights_only=True)
    n = ResNet(d["num_res_blocks"], d["num_hidden"], device)
    n.load_state_dict(d["state_dict"])
    return n


def train_new(ngames, device):
    model = ResNet(1, 1, device)
    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    train(
        model,
        optimiser,
        ngames,
        n_epochs=5,
        n_mcts_sims=5,
        device=device,
        mask_invalid_actions=True,
        train_batch_size=4,
    )
    return AlphaZeroAgent.from_nn(model, device=device, n_mcts_sims=10)


def train(
    model: nn.Module,
    optimiser,
    n_games: int,
    n_epochs: int,
    n_mcts_sims: int,
    device: str,
    mask_invalid_actions: bool,
    train_batch_size,
    verbose=True,
    parallel=False,
):
    """Self play n_games, train over the resulting data for n_epochs.
    Returns: avg_policy_loss, avg_value_loss
    """

    def mcts_eval(env):
        return _eval_for_mcts(model, env, device)

    def batch_mcts_eval(envs):
        return _batch_eval_for_mcts(model, envs, device)

    game_steps = []
    model.eval()
    with torch.no_grad():
        if parallel:
            game_steps = list(_self_play_n_games(batch_mcts_eval, n_games, n_mcts_sims))
        else:
            for _ in range(n_games):
                game_steps += _self_play_one_game(mcts_eval, n_mcts_sims)

    model.train()
    pls, vls = [], []
    for epoch in range(n_epochs):
        random.shuffle(game_steps)
        for batch in itertools.batched(game_steps, train_batch_size):
            pl, vl = _update_net(model, optimiser, batch, mask_invalid_actions, device)
            pls.append(pl)
            vls.append(vl)
        if verbose:
            print(
                f"epoch {epoch}: avg policy, value loss: {sum(pls) / len(pls):.4f}  {sum(vls) / len(vls):.4f}"
            )

    return sum(pls) / len(pls), sum(vls) / len(vls)


def _update_net(
    model: nn.Module,
    optimizer,
    game_steps: list[GameStep],
    mask_invalid_actions,
    device,
    fancy_loss=False,
    policy_loss_type: Literal["ce", "log sm kl"] = "ce",
    value_loss_type: Literal["mse", "huber"] = "mse",
    max_loss_coeff: float = 0.5,
):
    """Train the network over all given samples (steps) in one batch.
    Returns: policy loss, value loss
    """
    for s in game_steps:
        assert is_prob_dist(s.mcts_probs)
        for i, v in enumerate(s.valid_action_mask):
            if not v:
                assert s.board[i] != t3.EMPTY
                assert s.mcts_probs[i] == 0
    step_tuples = [s.as_tuple() for s in game_steps]
    *_, policy_targets, value_targets = zip(*step_tuples)
    state = torch.stack(
        [_board2tensor(board, player) for board, player, *_ in step_tuples]
    ).to(device)

    policy_targets, value_targets = (
        np.array(policy_targets),
        np.array(value_targets).reshape(-1, 1),
    )

    policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
    value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)

    out_policy, out_value = model(state)

    if mask_invalid_actions:
        valid_action_masks = torch.stack(
            [
                torch.tensor(s.valid_action_mask, dtype=torch.bool, device=device)
                for s in game_steps
            ]
        )
        out_policy = out_policy.masked_fill(~valid_action_masks, -1e32)

    if not fancy_loss:
        policy_loss = F.cross_entropy(out_policy, policy_targets)
        value_loss = F.mse_loss(out_value, value_targets)
        loss = policy_loss + value_loss
    else:
        if policy_loss_type == "ce":
            policy_loss_per_sample = F.cross_entropy(
                out_policy, policy_targets, reduction="none"
            )
            policy_loss = policy_loss_per_sample.mean()
        elif policy_loss_type == "log sm kl":
            log_probs = F.log_softmax(out_policy, dim=1)
            policy_loss_per_sample = F.kl_div(
                log_probs, policy_targets, reduction="none"
            )
            policy_loss = policy_loss_per_sample.mean()
        else:
            raise Exception("unknown policy loss type " + policy_loss_type)

        if value_loss_type == "mse":
            value_loss_per_sample = F.mse_loss(
                out_value, value_targets, reduction="none"
            )
            value_loss = value_loss_per_sample.mean()
        elif value_loss_type == "huber":
            value_loss_per_sample = F.huber_loss(
                out_value, value_targets, reduction="none"
            )
            value_loss = value_loss_per_sample.mean()
        else:
            raise Exception("unknown value loss type " + value_loss_type)

        total_loss_per_sample = policy_loss_per_sample + value_loss_per_sample.squeeze()
        max_loss = total_loss_per_sample.max()
        loss = policy_loss + value_loss + max_loss_coeff * max_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def _eval_for_mcts(
    model: nn.Module, env: t3.TttEnv, device
) -> tuple[list[float], float]:
    p, v = _net_fwd(model, env, device)
    return softmax(p.tolist()[0]), v.item()


def _batch_eval_for_mcts(
    model: nn.Module, envs: list[t3.TttEnv], device
) -> list[tuple[list[float], float]]:
    p, v = _net_fwd_batch(model, envs, device)
    p = p.softmax(dim=1).tolist()
    v = v.squeeze().tolist()
    if type(v) is float:
        v = [v]
    return list(zip(p, v))


def _mcts_probs(root: mcts.MCTSNode) -> list[float]:
    probs = [0] * 9
    total_visits = sum(c.visits for c in root.children.values())
    for a, n in root.children.items():
        probs[a] = n.visits / total_visits
    assert is_prob_dist(probs)
    return probs


def _valid_actions_mask(env: t3.TttEnv):
    valid_actions = list(env.valid_actions())
    return [x in valid_actions for x in range(9)]


def _self_play_one_game(eval_fn: mcts.EvaluateFunc, n_sims=10) -> list[GameStep]:
    """Plays one game
    Returns list[(board state, mcts probs, final reward for current player)]"""
    env = t3.TttEnv()
    mem = []
    while True:
        state = env.board[:]
        valid_mask = _valid_actions_mask(env)
        root = mcts.mcts_search(
            env, eval_fn, n_sims, c_puct=3.0, add_dirichlet_noise=True
        )
        probs = _mcts_probs(root)
        mem.append((state, env.current_player, valid_mask, probs))
        action = np.random.choice(9, p=probs)
        _, reward, game_over, _, _ = env.step(action)
        if game_over:
            return [
                GameStep(
                    state, player, mask, probs, reward if player == t3.X else -reward
                )
                for state, player, mask, probs in mem
            ]


def _self_play_n_games(
    eval_fn: mcts.BatchEvaluateFunc, n_games, n_mcts_sims
) -> typing.Iterable[GameStep]:
    envs = [t3.TttEnv() for _ in range(n_games)]
    game_overs = [False for _ in range(n_games)]
    trajectories = [[] for _ in range(n_games)]
    while not all(game_overs):
        active_idxs = [i for i, go in enumerate(game_overs) if not go]
        active_envs = [envs[i] for i in active_idxs]
        roots = mcts.mcts_search_parallel(
            active_envs,
            eval_fn,
            num_simulations=n_mcts_sims,
            c_puct=3.0,
            add_dirichlet_noise=True,
        )
        for i, root in zip(active_idxs, roots):
            env = root.state
            valid_mask = _valid_actions_mask(root.state)
            probs = _mcts_probs(root)
            trajectories[i].append(
                (env.board[:], env.current_player, valid_mask, probs)
            )
            action = np.random.choice(9, p=probs)
            _, reward, game_over, _, _ = env.step(action)
            if game_over:
                trajectories[i].append(reward)
                game_overs[i] = True
    for t in trajectories:
        reward = t[-1]
        assert reward in [-1, 0, 1]
        for state, player, mask, probs in t[:-1]:
            yield GameStep(
                state, player, mask, probs, reward if player == t3.X else -reward
            )


def _env2tensor(env):
    return _board2tensor(env.board, env.current_player)


def _board2tensor(board: t3.Board, current_player: t3.Player):
    # encodes a state in a player-independent way
    layers = (
        [c == current_player for c in board],
        [c == t3.EMPTY for c in board],
        [c == t3.other_player(current_player) for c in board],
    )
    tlayers = [torch.tensor(x, dtype=torch.float32).reshape((3, 3)) for x in layers]
    return torch.stack(tlayers)


# todo: think of a better name
def _net_fwd(model: nn.Module, env: t3.TttEnv, device):
    return _net_fwd_board(model, env.board, env.current_player, device)


def _net_fwd_batch(model: nn.Module, envs: list[t3.TttEnv], device):
    boards = [e.board for e in envs]
    cps = [e.current_player for e in envs]
    return _net_fwd_board_batch(model, boards, cps, device)


def _net_fwd_board_batch(
    model: nn.Module, boards: list[t3.Board], current_players: list[t3.Player], device
):
    batch = torch.stack(
        [_board2tensor(b, cp) for b, cp in zip(boards, current_players)]
    ).to(device)
    policy, val = model(batch)
    return policy, val


def _net_fwd_board(
    model: nn.Module, board: t3.Board, current_player: t3.Player, device
):
    enc_state = _board2tensor(board, current_player)
    minput = enc_state.unsqueeze(0).to(device)
    policy, val = model(minput)
    return policy, val


def nn_2_eval(model: nn.Module, device):
    def eval_fn(env: t3.TttEnv):
        return _eval_for_mcts(model, env, device)

    return eval_fn


def nn_2_batch_eval(model: nn.Module, device):
    def eval_fn(envs: list[t3.TttEnv]):
        return _batch_eval_for_mcts(model, envs, device)

    return eval_fn


class AlphaZeroAgent(TttAgent):
    def __init__(self, n_mcts_sims, mcts_eval, c_puct=1.0, batch_mcts_eval=None):
        self._mcts_eval = mcts_eval
        self._n_mcts_sims = n_mcts_sims
        self._c_puct = c_puct
        self._batch_mcts_eval = batch_mcts_eval

    @staticmethod
    def from_nn(model: nn.Module, n_mcts_sims: int, device: str):
        return AlphaZeroAgent(
            mcts_eval=nn_2_eval(model, device),
            n_mcts_sims=n_mcts_sims,
            batch_mcts_eval=nn_2_batch_eval(model, device),
        )

    @staticmethod
    def from_eval(mcts_eval, n_mcts_sims: int, c_puct=1.0):
        return AlphaZeroAgent(
            mcts_eval=mcts_eval, n_mcts_sims=n_mcts_sims, c_puct=c_puct
        )

    def get_action(self, env: t3.TttEnv):
        root = mcts.mcts_search(
            env, self._mcts_eval, self._n_mcts_sims, c_puct=self._c_puct
        )
        return max(
            root.children,
            key=lambda move: (root.children[move].visits, root.children[move].value()),
        )

    def get_actions(self, envs: list[t3.TttEnv]) -> list[int]:
        if self._batch_mcts_eval is None:
            raise Exception("need batch eval for get actions")
        roots = mcts.mcts_search_parallel(
            envs, self._batch_mcts_eval, self._n_mcts_sims, c_puct=self._c_puct
        )
        return [self.best_child(root) for root in roots]

    def save(self, path: str | Path):
        return NotImplementedError()

    def best_child(self, node: mcts.MCTSNode):
        return max(
            node.children,
            key=lambda move: (node.children[move].visits, node.children[move].value()),
        )
