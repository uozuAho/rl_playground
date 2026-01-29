"""AlphaZero-style MCTS search

This implements Monte Carlo Tree Search as used in AlphaZero, which uses:
- PUCT (Predictor + UCT) for tree policy instead of UCB1
- Neural network evaluation function that returns policy + value
- Policy priors guide the search
"""

from dataclasses import dataclass, field
from math import sqrt
import typing as t

import ttt.env as t3
from utils import maths
from utils.maths import is_prob_dist

"""
Evaluation function that takes a game state and returns:
- policy: probability distribution over actions (length 9 for tic-tac-toe)
- value: estimated value of the state for the current player (-1 to 1)
"""
type EvaluateFunc = t.Callable[[t3.TttEnv], tuple[list[float], float]]

type BatchEvaluateFunc = t.Callable[[list[t3.TttEnv]], list[tuple[list[float], float]]]


class MCTSNode:
    def __init__(
        self,
        state: t3.TttEnv,
        parent: "MCTSNode | None",
        prior: float,
        action_from_parent: int | None = None,
    ):
        self.state = state
        self.parent = parent
        self.prior = prior  # P(s,a) from policy network
        self.v_est = None  # V(s) from policy network
        self.action_from_parent = action_from_parent

        self.children: dict[int, MCTSNode] = {}  # action -> child node
        self.visits = 0  # N(s,a)
        self.total_value = 0.0  # W(s,a) - sum of values backed up through this node

    def __repr__(self):
        v_est_s = "None" if not self.v_est else f"{self.v_est:.2f}"
        board = self.state.str1d_sep("|")
        return (
            f"{board} vis: {self.visits}, puc1: {self.puct(1.0):.2f}, "
            f"P: {self.prior:.2f}, v: {self.value():.2f}, "
            f"tv: {self.total_value:.2f}, v_est: {v_est_s}"
        )

    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def puct(self, c_puct: float):
        v = 0 if not self.parent else sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + c_puct * self.prior * v

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        return self.state.status() != t3.IN_PROGRESS


def mcts_search(
    env: t3.TttEnv,
    evaluate: EvaluateFunc,
    num_simulations: int,
    c_puct: float = 1.0,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> MCTSNode:
    """
    Run AlphaZero MCTS search from the given state.

    Args:
        env: Game state to search from
        evaluate: Function that returns (policy, value) for a state
        num_simulations: Number of MCTS simulations to run
        c_puct: Exploration constant for PUCT formula (typically 1-5)
        add_dirichlet_noise: Whether to add Dirichlet noise to root priors (for exploration during training)
        dirichlet_alpha: Alpha parameter for Dirichlet noise
        dirichlet_epsilon: Weight of noise vs original prior

    Returns:
        Root node of the search tree with visit counts and values
    """
    root = MCTSNode(env, parent=None, prior=1.0)

    for _ in range(num_simulations):
        node = root

        # Selection: traverse tree using PUCT until we reach a leaf
        while node.is_expanded() and not node.is_terminal():
            node = max(node.children.values(), key=lambda c: c.puct(c_puct))

        if node.is_terminal():
            # Terminal node: use actual game outcome
            status = node.state.status()
            who_moved_last = t3.other_player(node.state.current_player)
            if status == who_moved_last:
                value = 1.0
            elif status == t3.DRAW:
                value = 0.0
            else:
                value = -1.0
        else:
            # Non-terminal leaf: evaluate
            policy, value = evaluate(node.state)
            # evaluate gives the value for the current player, we want
            # for the previous player - just need to invert the value
            value = -value
            node.v_est = value

            assert is_prob_dist(policy)

            if node == root and add_dirichlet_noise:
                policy = maths.add_dirichlet_noise(
                    policy, dirichlet_alpha, dirichlet_epsilon
                )

            assert is_prob_dist(policy)

            # Expand: create child nodes for all valid actions
            for action in node.state.valid_actions():
                child_state = node.state.copy()
                child_state.step(action)
                node.children[action] = MCTSNode(
                    state=child_state,
                    parent=node,
                    prior=policy[action],
                    action_from_parent=action,
                )

        # Backpropagation: update values up the search path
        while node:
            node.visits += 1
            node.total_value += value
            node = node.parent
            value = -value

    return root


def mcts_search_parallel(
    envs: list[t3.TttEnv],
    evaluate: BatchEvaluateFunc,
    num_simulations: int,
    c_puct: float = 1.0,
    add_dirichlet_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> list[MCTSNode]:
    pm = _ParallelMcts(
        envs,
        evaluate,
        num_simulations,
        c_puct,
        add_dirichlet_noise,
        dirichlet_alpha,
        dirichlet_epsilon,
    )
    return pm.run()


class _MctsSimState:
    def __init__(self, root: MCTSNode):
        self.root = root
        self.node = root
        self.terminal_value = None
        self.p_eval = None
        self.v_eval = None

    def reset(self):
        self.node = self.root
        self.terminal_value = None
        self.p_eval = None
        self.v_eval = None


@dataclass
class _ParallelMcts:
    envs: list[t3.TttEnv]
    evaluate: BatchEvaluateFunc
    num_simulations: int
    c_puct: float = 1.0
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    _sim_count = 0
    _sims: list[_MctsSimState] = field(default_factory=list)

    def run(self):
        self._sims = [
            _MctsSimState(MCTSNode(env, parent=None, prior=1.0)) for env in self.envs
        ]
        while self._sim_count < self.num_simulations:
            self._start_sim()
            self._eval()
            self._finish_sim()
            self._sim_count += 1
        return [s.root for s in self._sims]

    def _start_sim(self):
        for sim in self._sims:
            sim.reset()

            # Selection: traverse tree using PUCT until we reach a leaf
            while sim.node.is_expanded() and not sim.node.is_terminal():
                sim.node = max(
                    sim.node.children.values(), key=lambda c: c.puct(self.c_puct)
                )

            if sim.node.is_terminal():
                # Terminal node: use actual game outcome
                status = sim.node.state.status()
                who_moved_last = t3.other_player(sim.node.state.current_player)
                if status == who_moved_last:
                    sim.terminal_value = 1.0
                elif status == t3.DRAW:
                    sim.terminal_value = 0.0
                else:
                    sim.terminal_value = -1.0

    def _eval(self):
        envs = [s.node.state for s in self._sims]
        pvs = self.evaluate(envs)
        for i, pv in enumerate(pvs):
            p, v = pv
            self._sims[i].p_eval = p
            self._sims[i].v_eval = v

    def _finish_sim(self):
        for sim in self._sims:
            assert sim.p_eval is not None
            assert sim.v_eval is not None

            if sim.terminal_value is None:
                # evaluate gives the value for the current player, we want
                # for the previous player - just need to invert the value
                sim.v_eval = -sim.v_eval
                sim.node.v_est = sim.v_eval

                assert is_prob_dist(sim.p_eval)

                if sim.node == sim.root and self.add_dirichlet_noise:
                    sim.p_eval = maths.add_dirichlet_noise(
                        sim.p_eval, self.dirichlet_alpha, self.dirichlet_epsilon
                    )

                assert is_prob_dist(sim.p_eval)

                # Expand: create child nodes for all valid actions
                for action in sim.node.state.valid_actions():
                    child_state = sim.node.state.copy()
                    child_state.step(action)
                    sim.node.children[action] = MCTSNode(
                        state=child_state,
                        parent=sim.node,
                        prior=sim.p_eval[action],
                        action_from_parent=action,
                    )

            value = sim.terminal_value if sim.terminal_value else sim.v_eval
            assert value is not None
            assert value is not True

            # Backpropagation: update values up the search path
            while sim.node:
                sim.node.visits += 1
                sim.node.total_value += value
                sim.node = sim.node.parent
                value = -value
