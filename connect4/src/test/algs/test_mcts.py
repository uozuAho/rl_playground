from agents.mcts_agent import uniform_eval_batch
from algs.mcts import ParallelMcts, MCTSNode
import env.connect4 as c4
from utils.maths import is_prob_dist


def test_mcts_basic():
    num_games = 2
    num_sims = 3
    states = [c4.new_game() for _ in range(num_games)]

    roots = ParallelMcts(
        states,
        evaluate_fn=uniform_eval_batch,
        num_simulations=num_sims,
        c_puct=1.1,
        add_dirichlet_noise=True,
    ).run()

    assert len(roots) == num_games
    for root in roots:
        assert root.visits == num_sims
        for node in all_nodes(root):
            cpriors = [c.prior for c in node.children.values()]
            if cpriors:
                assert is_prob_dist(cpriors)


def test_chooses_best_moves():
    boards_best_moves = [
        (
            """.......
.......
.......
.......
OOO....
XXX....""",
            3,
        ),
        (
            """.......
.......
.......
.......
XXX....
OOO....""",
            3,
        ),
        (
            """.......
.......
.......
.......
XXX....
OOO...X""",
            3,
        ),
    ]

    states = [c4.from_string(x[0]) for x in boards_best_moves]
    expected_best_moves = [x[1] for x in boards_best_moves]

    roots = ParallelMcts(
        states,
        evaluate_fn=uniform_eval_batch,
        num_simulations=100,
        c_puct=1.1,
        add_dirichlet_noise=True,
    ).run()

    maxvisits = [max(root.children.values(), key=lambda c: c.visits) for root in roots]
    best_moves = [x.action_from_parent for x in maxvisits]
    assert best_moves == expected_best_moves


def all_nodes(node: MCTSNode | None):
    yield node
    for c in node.children.values():
        yield from all_nodes(c)
