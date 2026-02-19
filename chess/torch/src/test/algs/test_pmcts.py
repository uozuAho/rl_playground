import chess

from agents import mctsnew
from agents.mctsnew import uniform_eval_batch
from algs.pmcts import ParallelMcts, MCTSNode
from env import env
from utils.maths import is_prob_dist
from utils.play import play_games_parallel


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


def test_mctsu_should_be_stronger_with_more_sims():
    strong_agent = mctsnew.make_uniform_agent(20)
    weak_agent = mctsnew.make_uniform_agent(10)
    wld_strong_vs_weak = play_games_parallel(strong_agent, weak_agent, 50)
    wld_weak_vs_strong = play_games_parallel(weak_agent, strong_agent, 50)
    strong_wins = wld_strong_vs_weak[0] + wld_weak_vs_strong[1]
    strong_losses = wld_strong_vs_weak[1] + wld_weak_vs_strong[0]
    assert strong_wins > strong_losses


def test_chooses_best_moves():
    boards_best_moves = [
        ("6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1", "f1e1"),
    ]

    states = [env.ChessGame(fen=x[0]) for x in boards_best_moves]
    expected_best_moves = [chess.Move.from_uci(x[1]) for x in boards_best_moves]

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
    if node is not None:
        yield node
        for c in node.children.values():
            yield from all_nodes(c)
