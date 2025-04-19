from ttt.agents.mcts import MctsAgent, random_rollout_reward
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
