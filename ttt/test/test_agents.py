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
