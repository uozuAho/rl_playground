import algs.az_mcts as az_mcts
import algs.az_evaluators as az_eval
import ttt.env as t3


def test_basic_search():
    env = t3.TttEnv()
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=10,
    )

    assert root.visits == 10
    assert root.is_expanded()


def test_basic_parallel_search():
    envs = [t3.TttEnv() for _ in range(2)]
    roots = az_mcts.mcts_search_parallel(
        envs,
        evaluate=az_eval.uniform_batch,
        num_simulations=10,
    )

    for root in roots:
        assert root.visits == 10
        assert root.is_expanded()


def test_terminal_state_value():
    x_win = t3.TttEnv.from_str("xxx|oo.|o..")
    root = az_mcts.mcts_search(
        env=x_win,
        evaluate=az_eval.uniform,
        num_simulations=1,
    )
    assert root.visits == 1
    assert not root.is_expanded()  # Terminal, so not expanded


def test_node_value():
    """Test MCTSNode value calculation"""
    env = t3.TttEnv()
    node = az_mcts.MCTSNode(env, parent=None, prior=1.0)

    # Initially zero visits, value should be 0
    assert node.value() == 0.0

    # After some visits
    node.visits = 10
    node.total_value = 5.0
    assert node.value() == 0.5


def test_finds_winning_move():
    env = t3.TttEnv.from_str("xx.|oo.|...")
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.winning_move_evaluator,
        num_simulations=50,
    )

    winning_action = 2
    assert winning_action in root.children

    max_visits = max(child.visits for child in root.children.values())
    assert root.children[winning_action].visits == max_visits


def test_blocks_opponent_win():
    """Test that MCTS finds a blocking move for O"""
    env = t3.TttEnv.from_str("x..|oo.|x..")
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.winning_move_evaluator,
        num_simulations=50,
    )

    blocking_action = 5
    assert blocking_action in root.children

    max_visits = max(child.visits for child in root.children.values())
    assert root.children[blocking_action].visits == max_visits


def test_search_path_expansion():
    """Test that search expands nodes correctly"""
    env = t3.TttEnv()
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=1,
    )

    # Root should be expanded after 1 simulation
    assert root.is_expanded()
    assert len(root.children) == 9  # All 9 positions are valid initially


def test_backpropagation():
    env = t3.TttEnv()

    def always_1_eval(env: t3.TttEnv) -> tuple[list[float], float]:
        policy = [1.0 / 9] * 9
        value = 0.1
        return policy, value

    root = az_mcts.mcts_search(
        env=env,
        evaluate=always_1_eval,
        num_simulations=20,
    )

    assert all(child.visits > 0 for child in root.children.values())

    # Since evaluator returns +1 and values flip signs,
    # root should accumulate negative values (opponent's perspective)
    assert root.total_value < 0


def test_handles_near_terminal_state():
    # One move away from draw
    env = t3.TttEnv.from_str("xox|oxo|.xo")
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=5,
    )

    assert len(root.children) == 1


def test_dirichlet_noise():
    """Test that Dirichlet noise can be added (doesn't crash)"""
    env = t3.TttEnv()
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=10,
        add_dirichlet_noise=True,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    assert root.visits == 10
    assert root.is_expanded()


def test_parent_child_relationships():
    """Test that parent-child relationships are set up correctly"""
    env = t3.TttEnv()
    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=5,
    )

    # Check parent pointers
    assert root.parent is None
    for action, child in root.children.items():
        assert child.parent is root
        assert child.action_from_parent == action


def test_valid_actions_only():
    """Test that only valid actions are expanded"""
    # Create a state with limited valid moves
    env = t3.TttEnv.from_str("xxoo.x.o.")
    valid = list(env.valid_actions())

    root = az_mcts.mcts_search(
        env=env,
        evaluate=az_eval.uniform,
        num_simulations=10,
    )

    # Only valid actions should be in children
    assert len(root.children) == len(valid)
    assert all(action in valid for action in root.children.keys())
