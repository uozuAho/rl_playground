import pytest

from agents.mcts_agent import uniform_eval_batch, make_uniform_agent
from algs.mcts import ParallelMcts, MCTSNode
import env.connect4 as c4
from utils.maths import is_prob_dist
from utils.play import play_games_parallel


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


@pytest.mark.skip("fix this after fixing specific test below")
def test_mctsu_should_be_stronger_with_more_sims():
    strong_agent = make_uniform_agent(20)
    weak_agent = make_uniform_agent(10)
    wld_strong_vs_weak = play_games_parallel(strong_agent, weak_agent, 50)
    wld_weak_vs_strong = play_games_parallel(weak_agent, strong_agent, 50)
    strong_wins = wld_strong_vs_weak[0] + wld_weak_vs_strong[1]
    strong_losses = wld_strong_vs_weak[1] + wld_weak_vs_strong[0]
    assert strong_wins > strong_losses


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


def test_specific_case():
    states = [
        # todo: dedent in from_string
        c4.from_string(
            """OOXX...
XXOOXO.
OOXXOX.
XXOOXX.
OOXOOXO
XXXOXOO"""
        )
    ]

    agent_x = make_uniform_agent(10)
    agent_o = make_uniform_agent(20)

    while not states[0].done:
        # todo: make single game version for convenience
        agent = agent_x if states[0].current_player == c4.PLAYER1 else agent_o
        actions = agent.get_actions(states)
        states = [c4.make_move(s, a) for s, a in zip(states, actions)]


def all_nodes(node: MCTSNode | None):
    yield node
    for c in node.children.values():
        yield from all_nodes(c)
