import chess

from agents.mctsnew import uniform_eval_batch, random_rollout_eval_batch
from algs.pmcts import ParallelMcts, MCTSNode
from env import env
from utils.maths import is_prob_dist


def test_mcts_basic():
    num_games = 2
    num_sims = 3
    states = [env.ChessGame() for _ in range(num_games)]

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
        ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8"),
        ("4r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 1 1", "e8e1"),
    ]

    states = [env.ChessGame(fen=x[0]) for x in boards_best_moves]
    expected_best_moves = [chess.Move.from_uci(x[1]) for x in boards_best_moves]

    roots = ParallelMcts(
        states,
        evaluate_fn=random_rollout_eval_batch,
        num_simulations=100,
        c_puct=1.0,
        add_dirichlet_noise=False,
    ).run()

    maxvisits = [max(root.children.values(), key=lambda c: c.visits) for root in roots]
    best_moves = [x.action_from_parent for x in maxvisits]
    assert best_moves == expected_best_moves


def all_nodes(node: MCTSNode | None):
    if node is not None:
        yield node
        for c in node.children.values():
            yield from all_nodes(c)
