import agents.mcts as mcts
import ttt.env as t3


def test_plays_constant_val():
    env = t3.TttEnv()
    agent = mcts.MctsAgent(n_sims=1, valfn=lambda x, y: 1)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)


def test_plays_with_random_rollout():
    env = t3.TttEnv()
    agent = mcts.MctsAgent(n_sims=1, valfn=mcts.random_rollout_reward)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)


def test_random_rollout_reward():
    x_win = t3.TttEnv.from_str("xxx|oo.|o..")
    o_win = t3.TttEnv.from_str("ooo|xx.|x..")
    assert mcts.random_rollout_reward(x_win, t3.X) == 1
    assert mcts.random_rollout_reward(x_win, t3.O) == -1
    assert mcts.random_rollout_reward(o_win, t3.X) == -1
    assert mcts.random_rollout_reward(o_win, t3.O) == 1


def test_x_should_block_win():
    env = t3.TttEnv.from_str("..x|.oo|..x")
    agent = mcts.MctsAgent(n_sims=100, valfn=mcts.random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3


def test_o_should_block_win():
    env = t3.TttEnv.from_str("x.o.xx..o")
    agent = mcts.MctsAgent(n_sims=100, valfn=mcts.random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3


def test_x_should_win():
    env = t3.TttEnv.from_str("x.o.xx.oo")
    agent = mcts.MctsAgent(n_sims=30, valfn=mcts.random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3


def test_o_should_win():
    env = t3.TttEnv.from_str("oxx.oo.xx")
    agent = mcts.MctsAgent(n_sims=30, valfn=mcts.random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3
