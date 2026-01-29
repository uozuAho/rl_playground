from agents.qlearn import TabQlearnAgent
from agents.sarsa import TabSarsaAgent
from agents.tab_greedy_v import TabGreedyVAgent, StateValueTable
from agents.random import RandomAgent
from agents.perfect import PerfectAgent
import ttt.env as t3


def test_perfect_x_agent_never_loses_to_random():
    p_agent = PerfectAgent()
    r_agent = RandomAgent()
    env = t3.TttEnv()
    for _ in range(1000):
        env.reset()
        while env.status() == t3.IN_PROGRESS:
            if env.current_player == t3.X:
                action = p_agent.get_action(env)
            else:
                action = r_agent.get_action(env)
            env.step(action)
        assert env.status() in [t3.DRAW, t3.X]


def test_perfect_o_agent_never_loses_to_random():
    r_agent = RandomAgent()
    p_agent = PerfectAgent()
    env = t3.TttEnv()
    for _ in range(1000):
        actions = []
        env.reset()
        while env.status() == t3.IN_PROGRESS:
            if env.current_player == t3.X:
                action = r_agent.get_action(env)
            else:
                action = p_agent.get_action(env)
            actions.append(("x" if env.current_player == t3.X else "o", action))
            env.step(action)
        if env.status() not in [t3.DRAW, t3.O]:
            for a in actions:
                print(a)
            print(env.str2d())
            raise Exception("perfect o lost")


def test_perfect_agents_always_draw():
    x_agent = PerfectAgent()
    o_agent = PerfectAgent()
    env = t3.TttEnv()
    for _ in range(20):
        env.reset()
        while env.status() == t3.IN_PROGRESS:
            action = (
                x_agent.get_action(env)
                if env.current_player == t3.X
                else o_agent.get_action(env)
            )
            env.step(action)
        assert env.status() == t3.DRAW


def test_sarsa_train_and_play():
    agent = TabSarsaAgent()
    env = t3.TttEnv()
    agent.train(env, RandomAgent(), 1)
    env.reset()
    done = False
    while not done:
        action = agent.get_action(env)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc


def test_qlearn_train_and_play():
    agent = TabQlearnAgent()
    env = t3.TttEnv()
    agent.train(env, RandomAgent(), 1)
    env.reset()
    done = False
    while not done:
        action = agent.get_action(env)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc


def test_greedy_v_is_greedy():
    qtable = {
        "....x....": 0.9,
    }
    agent = TabGreedyVAgent(StateValueTable.from_dict(qtable))
    env = t3.TttEnv()
    assert agent.get_action(env) == 4
