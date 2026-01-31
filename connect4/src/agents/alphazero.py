import itertools
import random
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import env.connect4 as c4
from agents import mcts_agent
from agents.az_nets import ResNet
from algs import mcts
from utils import maths, types


@dataclass
class GameStep:
    state: c4.GameState
    valid_action_mask: np.ndarray
    mcts_probs: np.ndarray
    final_value: float

    def as_tuple(self):
        return (
            self.state,
            self.valid_action_mask,
            self.mcts_probs,
            self.final_value,
        )

    def __repr__(self):
        b = "todo board string"
        p = "X" if self.state.current_player == c4.PLAYER1 else "O"
        m = ",".join(f"{x:.2f}" for x in self.mcts_probs)
        # TODO: valid action masks
        return f"{b} {p} {self.final_value:0.2f}  [{m}]"


def load_net(path: Path, device):
    d = torch.load(path, weights_only=True)
    n = ResNet(d["num_res_blocks"], d["num_hidden"], device)
    n.load_state_dict(d["state_dict"])
    return n


def train(
    model: nn.Module,
    optimiser,
    n_games: int,
    n_epochs: int,
    n_mcts_sims: int,
    device: str,
    mask_invalid_actions: bool,
    train_batch_size: int,
    verbose=True,
):
    """Self play n_games, train over the resulting data for n_epochs.
    Returns: avg_policy_loss, avg_value_loss
    """

    def batch_mcts_eval(envs):
        return _batch_eval_for_mcts(model, envs, device)

    game_steps = []
    model.eval()
    with torch.no_grad():
        game_steps = list(_self_play_n_games(batch_mcts_eval, n_games, n_mcts_sims))

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
    mask_invalid_actions: bool,
    device,
):
    """Train the network over all given samples (steps) in one batch.
    Returns: policy loss, value loss
    """
    for s in game_steps:
        assert maths.is_prob_dist(s.mcts_probs)
        for i, v in enumerate(s.valid_action_mask):
            if not v:
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

    policy_loss = F.cross_entropy(out_policy, policy_targets)
    value_loss = F.mse_loss(out_value, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def _batch_eval_for_mcts(
    model: nn.Module, states: list[c4.GameState], device
) -> list[tuple[list[float], float]]:
    p, v = _net_fwd_batch(model, states, device)
    p = p.softmax(dim=1).tolist()
    v = v.squeeze().tolist()
    if type(v) is float:
        v = [v]
    return list(zip(p, v))


def _mcts_probs(root: mcts.MCTSNode) -> list[float]:
    probs = [0] * c4.ACTION_SIZE
    total_visits = sum(c.visits for c in root.children.values())
    for a, n in root.children.items():
        probs[a] = n.visits / total_visits
    assert maths.is_prob_dist(probs)
    return probs


def _valid_actions_mask(state: c4.GameState):
    valid_actions = list(c4.get_valid_moves(state))
    return [x in valid_actions for x in range(c4.ACTION_SIZE)]


def _self_play_n_games(
    eval_fn: types.BatchEvaluateFunc,
    n_games: int,
    n_mcts_sims: int,
) -> typing.Iterable[GameStep]:
    states = [c4.new_game() for _ in range(n_games)]
    game_overs = [False for _ in range(n_games)]
    trajectories = [[] for _ in range(n_games)]
    winners: list[bool | None | c4.Player] = [False for _ in range(n_games)]
    while not all(game_overs):
        active_idxs = [i for i, go in enumerate(game_overs) if not go]
        active_envs = [states[i] for i in active_idxs]
        roots = mcts.ParallelMcts(
            active_envs,
            eval_fn,
            num_simulations=n_mcts_sims,
            c_puct=3.0,
            add_dirichlet_noise=True,
        ).run()
        for i, root in zip(active_idxs, roots):
            state = root.state
            valid_mask = _valid_actions_mask(root.state)
            probs = _mcts_probs(root)
            trajectories[i].append((state, valid_mask, probs))
            action = np.random.choice(c4.ACTION_SIZE, p=probs)
            new_state = c4.make_move(state, action)
            states[i] = new_state
            if new_state.done:
                game_overs[i] = True
                winners[i] = new_state.winner
    assert not any(x is False for x in winners)
    for i, t in enumerate(trajectories):
        end_state = t[-1]
        assert end_state.done
        for state, mask, probs in t:
            winner = winners[i]
            final_reward = (
                0 if winner is None else 1 if state.current_player == winner else -1
            )
            yield GameStep(state, mask, probs, final_reward)


def _state2tensor(state: c4.GameState):
    return _board2tensor(state.board, state.current_player)


def _board2tensor(board: np.ndarray, current_player: c4.Player):
    # encodes a state in a player-independent way
    layers = np.stack(
        board == -1,
        board == 0,
        board == 1,
    ).astype(np.float32)
    tlayers = [torch.tensor(x, dtype=torch.float32) for x in layers]
    return torch.stack(tlayers)


def _net_fwd(model: nn.Module, env: c4.GameState, device):
    return _net_fwd_board(model, env.board, env.current_player, device)


def _net_fwd_batch(model: nn.Module, envs: list[c4.GameState], device):
    boards = [e.board for e in envs]
    cps = [e.current_player for e in envs]
    return _net_fwd_board_batch(model, boards, cps, device)


def _net_fwd_board_batch(
    model: nn.Module, boards: list[np.ndarray], current_players: list[c4.Player], device
):
    batch = torch.stack(
        [_board2tensor(b, cp) for b, cp in zip(boards, current_players)]
    ).to(device)
    policy, val = model(batch)
    return policy, val


def _net_fwd_board(
    model: nn.Module, board: np.ndarray, current_player: c4.Player, device
):
    enc_state = _board2tensor(board, current_player)
    minput = enc_state.unsqueeze(0).to(device)
    policy, val = model(minput)
    return policy, val


def nn_2_batch_eval(model: nn.Module, device):
    def eval_fn(envs: list[c4.GameState]):
        return _batch_eval_for_mcts(model, envs, device)

    return eval_fn


def make_az_agent(model: nn.Module, n_sims: int, device):
    return mcts_agent.MctsAgent(
        batch_eval_fn=nn_2_batch_eval(model, device),
        select_action_fn=mcts_agent.best_by_visit_value,
        n_sims=n_sims,
    )
