from ttt.agents.qlearn import QlearnAgent
from ttt.agents.sarsa import SarsaAgent
from ttt.agents.tab_greedy_v import GreedyVAgent
import ttt.agents.tab_greedy_v
import ttt.env
from ttt.env import TicTacToeEnv
from ttt.agents.random import RandomAgent
from ttt.agents.perfect import PerfectAgent


def test_perfect_agent_never_loses_to_random():
    p_agent = PerfectAgent('X')
    r_agent = RandomAgent()
    env = TicTacToeEnv(opponent=r_agent)
    for _ in range(20):
        env.reset()
        while not env.is_game_over:
            action = p_agent.get_action(env)
            env.step(action)
        assert env.get_status() in [ttt.env.DRAW, ttt.env.X_WIN]


def test_perfect_agents_always_draw():
    x_agent = PerfectAgent('X')
    o_agent = PerfectAgent('O')
    env = TicTacToeEnv(opponent=o_agent)
    for _ in range(20):
        env.reset()
        while not env.is_game_over:
            env.render()
            action = x_agent.get_action(env)
            print('x action: ', action)
            env.step(action)
        assert env.get_status() == ttt.env.DRAW


def test_sarsa_train_and_play():
    for ye in [True, False]:
        agent = SarsaAgent(allow_invalid_actions=ye)
        env = TicTacToeEnv(
            opponent=RandomAgent(),
            on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER if ye else ttt.env.INVALID_ACTION_THROW)
        agent.train(env, 1)
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc


def test_qlearn_train_and_play():
    for ye in [True, False]:
        agent = QlearnAgent(allow_invalid_actions=ye)
        env = TicTacToeEnv(
            opponent=RandomAgent(),
            on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER if ye else ttt.env.INVALID_ACTION_THROW)
        agent.train(env, 1)
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
    agent = GreedyVAgent(ttt.agents.tab_greedy_v.Qtable(qtable))
    env = TicTacToeEnv()
    assert agent.get_action(env) == 4
