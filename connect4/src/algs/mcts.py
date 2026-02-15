import math
from dataclasses import dataclass, field

import env.connect4 as c4
from utils import maths, types


class MCTSNode:
    def __init__(
        self,
        state: c4.GameState | None,
        parent: "MCTSNode | None",
        prior: float,
        action_from_parent: int | None = None,
    ):
        self._state = state
        self.parent = parent
        self.prior = prior  # P(s,a) from policy network
        self.v_est = None  # V(s) from policy network
        self.action_from_parent = action_from_parent

        # valid action -> child node
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0  # N(s,a)
        self.total_value = 0.0  # W(s,a) - sum of values backed up through this node

    def __repr__(self):
        v_est_s = "None" if not self.v_est else f"{self.v_est:.2f}"
        board = "todo board str"
        return (
            f"{board} vis: {self.visits}, puc1: {self.puct(1.0):.2f}, "
            f"P: {self.prior:.2f}, v: {self.value():.2f}, "
            f"tv: {self.total_value:.2f}, v_est: {v_est_s}"
        )

    @property
    def state(self):
        """Lazily calculate state (for perf reasons)"""
        if self._state is None:
            self._state = c4.make_move(self.parent.state, self.action_from_parent)
        return self._state

    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def puct(self, c_puct: float):
        v = 0 if not self.parent else math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.value() + c_puct * self.prior * v

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        return self.state.done


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
class ParallelMcts:
    states: list[c4.GameState]
    evaluate_fn: types.BatchEvaluateFunc
    num_simulations: int
    c_puct: float = 1.0
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    _sim_count = 0
    _sims: list[_MctsSimState] = field(default_factory=list)

    def run(self):
        self._sims = [
            _MctsSimState(MCTSNode(env, parent=None, prior=1.0)) for env in self.states
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
                winner = sim.node.state.winner
                who_moved_last = c4.other_player(sim.node.state.current_player)
                if winner == who_moved_last:
                    sim.terminal_value = 1.0
                elif winner is None:
                    sim.terminal_value = 0.0
                else:
                    sim.terminal_value = -1.0

    def _eval(self):
        envs = [s.node.state for s in self._sims]
        pvs = self.evaluate_fn(envs)
        for i, pv in enumerate(pvs):
            p, v = pv
            self._sims[i].p_eval = p
            self._sims[i].v_eval = v

    def _finish_sim(self):
        for sim in self._sims:
            if sim.terminal_value is None:
                # evaluate gives the value for the current player, we want
                # for the previous player - just need to invert the value
                sim.v_eval = -sim.v_eval
                sim.node.v_est = sim.v_eval

                if sim.node == sim.root and self.add_dirichlet_noise:
                    sim.p_eval = maths.add_dirichlet_noise(
                        sim.p_eval, self.dirichlet_alpha, self.dirichlet_epsilon
                    )

                # Expand: create child nodes for all valid actions
                for action in c4.get_valid_moves(sim.node.state):
                    sim.node.children[action] = MCTSNode(
                        state=None,
                        parent=sim.node,
                        prior=sim.p_eval[action],
                        action_from_parent=action,
                    )

            value = sim.terminal_value if sim.terminal_value else sim.v_eval

            # Backpropagation: update values up the search path
            while sim.node:
                sim.node.visits += 1
                sim.node.total_value += value
                sim.node = sim.node.parent
                value = -value
