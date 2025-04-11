from ttt.agents.qlearn import TabQlearnAgent
from ttt.agents.sarsa import TabSarsaAgent
from ttt.agents.tab_greedy_v import TabGreedyVAgent
import ttt.agents.tab_greedy_v
from ttt.agents.random import RandomAgent
from ttt.agents.perfect import PerfectAgent
import ttt.env


def test_perfect_x_agent_never_loses_to_random():
    p_agent = PerfectAgent()
    r_agent = RandomAgent()
    env = ttt.env.Env()
    for _ in range(1000):
        env.reset()
        while env.status() == ttt.env.IN_PROGRESS:
            if env.current_player == ttt.env.X:
                action = p_agent.get_action(env)
            else:
                action = r_agent.get_action(env)
            env.step(action)
        assert env.status() in [ttt.env.DRAW, ttt.env.X]


def test_perfect_o_agent_never_loses_to_random():
    r_agent = RandomAgent()
    p_agent = PerfectAgent()
    env = ttt.env.Env()
    for _ in range(1000):
        actions = []
        env.reset()
        while env.status() == ttt.env.IN_PROGRESS:
            if env.current_player == ttt.env.X:
                action = r_agent.get_action(env)
            else:
                action = p_agent.get_action(env)
            actions.append(('x' if env.current_player == ttt.env.X else 'o', action))
            env.step(action)
        if env.status() not in [ttt.env.DRAW, ttt.env.O]:
            for a in actions:
                print(a)
            print(env.str2d())
            raise Exception("perfect o lost")


def test_perfect_agents_always_draw():
    x_agent = PerfectAgent()
    o_agent = PerfectAgent()
    env = ttt.env.Env()
    for _ in range(20):
        env.reset()
        while env.status() == ttt.env.IN_PROGRESS:
            action = x_agent.get_action(env) if env.current_player == ttt.env.X else o_agent.get_action(env)
            env.step(action)
        assert env.status() == ttt.env.DRAW


def test_sarsa_train_and_play():
    for ye in [True, False]:
        agent = TabSarsaAgent(allow_invalid_actions=ye)
        env = ttt.env.Env(
            invalid_action_response=ttt.env.INVALID_ACTION_GAME_OVER if ye else ttt.env.INVALID_ACTION_THROW)
        agent.train(env, RandomAgent(), 1)
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc


def test_qlearn_train_and_play():
    for ye in [True, False]:
        agent = TabQlearnAgent(allow_invalid_actions=ye)
        env = ttt.env.Env(invalid_action_response=ttt.env.INVALID_ACTION_GAME_OVER if ye else ttt.env.INVALID_ACTION_THROW)
        agent.train(env, RandomAgent(), 1)
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc


def test_greedy_v_is_greedy():
    qtable = {
        '...|.x.|...': 0.9,
    }
    agent = TabGreedyVAgent(ttt.agents.tab_greedy_v.Qtable(qtable))
    env = ttt.env.Env()
    assert agent.get_action(env) == 4
