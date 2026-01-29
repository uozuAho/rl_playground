import random

import numpy as np

import ttt.env as t3


def test_fuzz_full():
    for env in [t3.GymEnv(), t3.TttEnv()]:
        for _ in range(100):
            env.reset()
            done = False
            i = 0
            while not done:
                num_empty = sum(1 if c == t3.EMPTY else 0 for c in env.board)
                va = list(env.valid_actions())
                assert len(va) == num_empty
                for a in va:
                    assert env.board[a] == t3.EMPTY
                action = random.choice(va)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if isinstance(env, t3.GymEnv):
                    assert np.array_equal(obs, np.array(env.board).reshape((3, 3)))
                numx = sum(1 if c == t3.X else 0 for c in env.board)
                numo = sum(1 if c == t3.O else 0 for c in env.board)
                assert abs(numx - numo) <= 1
                assert numx <= 5
                assert numo <= 5
                assert 9 - numx - numo == num_empty - 1
                assert i < 10
                i += 1
                if not done:
                    assert reward == 0
            s = t3.status(env.board)
            assert s != 0
            if s == t3.O:
                assert reward == -1
            elif s == t3.X:
                assert reward == 1
            else:
                assert reward == 0


def test_str1d():
    env = t3.TttEnv()
    env.step(0)
    assert env.str1d() == "x........"
    env.step(1)
    assert env.str1d() == "xo......."
