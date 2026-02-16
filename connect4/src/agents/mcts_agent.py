import typing as t
from dataclasses import dataclass

import env.connect4 as c4
from algs import mcts
from algs.mcts import MCTSNode
from utils import types


type SelectActionFn = t.Callable[[MCTSNode], c4.Action]


# not the same as random rollout: value is uniform across all actions
def make_uniform_agent(n_sims: int):
    return MctsAgent(
        batch_eval_fn=uniform_eval_batch,
        select_action_fn=best_by_visit_value,
        n_sims=n_sims,
    )


@dataclass
class MctsAgent:
    batch_eval_fn: types.BatchEvaluateFunc
    select_action_fn: SelectActionFn
    n_sims: int
    c_puct: float = 1.0
    add_dirichlet_noise: bool = False
    alpha: float = 0.3
    epsilon: float = 0.25

    def get_action(self, state: c4.GameState):
        return self.get_actions([state])[0]

    def get_actions(self, states: list[c4.GameState]):
        roots = mcts.ParallelMcts(
            states,
            evaluate_fn=self.batch_eval_fn,
            num_simulations=self.n_sims,
            c_puct=self.c_puct,
            add_dirichlet_noise=self.add_dirichlet_noise,
            dirichlet_alpha=self.alpha,
            dirichlet_epsilon=self.epsilon,
        ).run()
        return [self.select_action_fn(r) for r in roots]


def uniform_eval_batch(states: list[c4.GameState]) -> list[types.PV]:
    return [([1.0 / c4.COLS] * c4.COLS, 0.0) for _ in range(len(states))]


def best_by_visit_value(node: mcts.MCTSNode):
    return max(
        node.children,
        key=lambda move: (node.children[move].visits, node.children[move].value()),
    )
