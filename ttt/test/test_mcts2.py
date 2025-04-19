from ttt.agents.mcts2 import MctsAgent2, random_rollout_reward
import ttt.env as t3


def test_plays_constant_val():
    env = t3.Env()
    agent = MctsAgent2(n_sims=1, valfn=lambda b: 1)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)


def test_plays_with_random_rollout():
    env = t3.Env()
    agent = MctsAgent2(n_sims=1, valfn=random_rollout_reward)
    while env.status() == t3.IN_PROGRESS:
        action = agent.get_action(env)
        env.step(action)
