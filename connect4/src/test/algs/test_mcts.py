from agents.mcts_agent import uniform_eval_batch
from algs.mcts import ParallelMcts, MCTSNode
import env.connect4 as c4
from utils.maths import is_prob_dist


def test_mcts():
    num_games = 2
    states = [c4.new_game() for _ in range(num_games)]

    roots = ParallelMcts(
        states,
        evaluate_fn=uniform_eval_batch,
        num_simulations=3,
        c_puct=1.1,
        add_dirichlet_noise=True,
    ).run()

    assert len(roots) == num_games
    for root in roots:
        for node in all_nodes(root):
            cpriors = [c.prior for c in node.children.values()]
            if cpriors:
                assert is_prob_dist(cpriors)


def all_nodes(node: MCTSNode | None):
    yield node
    for c in node.children.values():
        yield from all_nodes(c)
