import itertools
import random
import typing
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import env.connect4 as c4
from agents import mcts_agent
from agents.az_nets import AzNet
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
        p = "X" if self.state.current_player == c4.PLAYER1 else "O"
        b = "|".join(
            "".join("." if c == 0 else "X" if c == c4.PLAYER1 else "O" for c in row)
            for row in self.state.board
        )
        m = ",".join(f"{x:.2f}" for x in self.mcts_probs)
        # TODO: valid action masks
        return f"{p} {b} {self.final_value:0.2f}  [{m}]"


def train(
    net: AzNet,
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
        return _batch_eval_for_mcts(net, envs, device)

    game_steps = []
    net.eval()
    with torch.no_grad():
        game_steps = list(
            _self_play_n_games(
                batch_mcts_eval,
                n_games,
                n_mcts_sims,
                c_puct=2,
                temperature=1.25,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.125,
            )
        )

    net.train()
    pls, vls = [], []
    for epoch in range(n_epochs):
        random.shuffle(game_steps)
        for batch in itertools.batched(game_steps, train_batch_size):
            pl, vl = _update_net(net, optimiser, batch, mask_invalid_actions, device)
            pls.append(pl)
            vls.append(vl)
        if verbose:
            print(
                f"epoch {epoch}: avg policy, value loss: {sum(pls) / len(pls):.4f}  {sum(vls) / len(vls):.4f}"
            )

    return sum(pls) / len(pls), sum(vls) / len(vls)


def _update_net(
    net: AzNet,
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

    states = [s.state for s in game_steps]
    step_tuples = [s.as_tuple() for s in game_steps]
    *_, policy_targets, value_targets = zip(*step_tuples)
    policy_targets, value_targets = (
        np.array(policy_targets),
        np.array(value_targets).reshape(-1, 1),
    )
    policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
    value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)

    plogits, values = net.forward_batch(states)

    if mask_invalid_actions:
        valid_action_masks = torch.stack(
            [
                torch.tensor(s.valid_action_mask, dtype=torch.bool, device=device)
                for s in game_steps
            ]
        )
        plogits = plogits.masked_fill(~valid_action_masks, -1e32)

    policy_loss = F.cross_entropy(plogits, policy_targets)
    value_loss = F.mse_loss(values, value_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def _batch_eval_for_mcts(
    net: AzNet, states: list[c4.GameState], device
) -> list[tuple[list[float], float]]:
    return net.pv_batch(states)


def _mcts_probs(root: mcts.MCTSNode) -> list[float]:
    probs = [0] * c4.ACTION_SIZE
    total_visits = sum(c.visits for c in root.children.values())
    assert total_visits > 0
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
    c_puct: float,
    temperature: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> typing.Iterable[GameStep]:
    # todo: use temperature
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
            c_puct=c_puct,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
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
        for state, mask, probs in t:
            winner = winners[i]
            final_reward = (
                0 if winner is None else 1 if state.current_player == winner else -1
            )
            yield GameStep(state, mask, probs, final_reward)


def nn_2_batch_eval(net: AzNet, device):
    def eval_fn(states: list[c4.GameState]):
        return _batch_eval_for_mcts(net, states, device)

    return eval_fn


def make_az_agent(net: AzNet, n_sims: int, c_puct: float, device):
    return mcts_agent.MctsAgent(
        batch_eval_fn=nn_2_batch_eval(net, device),
        select_action_fn=mcts_agent.best_by_visit_value,
        n_sims=n_sims,
        c_puct=c_puct,
    )
