from ttt.agents.qlearn import TabQlearnAgent
from ttt.agents.sarsa import TabSarsaAgent
from ttt.agents.tab_greedy_v import TabGreedyVAgent
import ttt.agents.tab_greedy_v
import ttt.env
from ttt.env import TicTacToeEnv
from ttt.agents.random import RandomAgent
from ttt.agents.perfect import PerfectAgent
import ttt.env2 as ttt


def test_perfect_agent_never_loses_to_random():
    p_agent = PerfectAgent()
    r_agent = RandomAgent()
    env = ttt.Env()
    for _ in range(20):
        env.reset()
        while not env.is_game_over:
            if env.current_player == ttt.X:
                action = p_agent.get_action(env)
            else:
                action = r_agent.get_action(env)
            env.step(action)
        assert ttt.status() in [ttt.DRAW, ttt.X]


def test_perfect_agents_always_draw():
    x_agent = PerfectAgent()
    o_agent = PerfectAgent()
    env = ttt.Env()
    for _ in range(20):
        env.reset()
        while not env.is_game_over:
            action = x_agent.get_action(env) if env.current_player == ttt.X else o_agent.get_action(env)
            env.step(action)
        assert env.get_status() == ttt.DRAW


def test_sarsa_train_and_play():
    for ye in [True, False]:
        agent = TabSarsaAgent(allow_invalid_actions=ye)
        env = ttt.Env(
            invalid_action_response=ttt.INVALID_ACTION_GAME_OVER if ye else ttt.INVALID_ACTION_THROW)
        agent.train(env, 1)
        env.reset()
        done = False
        while not done:
            action = agent.get_action(env)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc


def test_qlearn_train_and_play():
    for ye in [True, False]:
        agent = TabQlearnAgent(allow_invalid_actions=ye)
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
    agent = TabGreedyVAgent(ttt.agents.tab_greedy_v.Qtable(qtable))
    env = TicTacToeEnv()
    assert agent.get_action(env) == 4
