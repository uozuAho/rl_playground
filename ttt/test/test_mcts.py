from ttt.agents.mcts import _MCTSNode, MctsAgent, _mcts_decision, random_rollout_reward
import ttt.env as t3


def test_plays_constant_val():
    env = t3.Env()
    agent = MctsAgent(n_sims=1, valfn=lambda x,y : 1)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)


def test_plays_with_random_rollout():
    env = t3.Env()
    agent = MctsAgent(n_sims=1, valfn=random_rollout_reward)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)


def test_random_rollout_reward():
    x_win = t3.Env.from_str('xxxoo.o..')
    o_win = t3.Env.from_str('oooxx.x..')
    assert random_rollout_reward(x_win, t3.X) == 1
    assert random_rollout_reward(x_win, t3.O) == -1
    assert random_rollout_reward(o_win, t3.X) == -1
    assert random_rollout_reward(o_win, t3.O) == 1


def test_o_should_block_win():
    env = t3.FastEnv.from_str('x.o.xx..o')
    agent = MctsAgent(n_sims=30, valfn=random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3


def test_x_should_win():
    env = t3.FastEnv.from_str('x.o.xx.oo')
    agent = MctsAgent(n_sims=30, valfn=random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3


def test_o_should_win():
    env = t3.FastEnv.from_str('oxx.oo.xx')
    agent = MctsAgent(n_sims=30, valfn=random_rollout_reward)
    action = agent.get_action(env)
    assert action == 3

# todo: get rid of this
def test_ucb1():
    for v in range(1, 4):
        for pv in range(1, 4):
            for t in range(-4, 4):
                p = _MCTSNode(None, None)
                n = _MCTSNode(None, p)
                p.visits = pv
                n.visits = v
                n.total_reward = t
                print(n)
