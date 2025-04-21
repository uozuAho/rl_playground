import random

import numpy as np

import ttt.env


def test_fuzz_full():
    for env in [ttt.env.Env(), ttt.env.FastEnv()]:
        for _ in range(100):
            env.reset()
            done = False
            i = 0
            while not done:
                num_empty = sum(1 if c == ttt.env.EMPTY else 0 for c in env.board)
                va = list(env.valid_actions())
                assert len(va) == num_empty
                for a in va:
                    assert env.board[a] == ttt.env.EMPTY
                action = random.choice(va)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if isinstance(env, ttt.env.Env):
                    assert np.array_equal(obs, np.array(env.board).reshape((3,3)))
                numx = sum(1 if c == ttt.env.X else 0 for c in env.board)
                numo = sum(1 if c == ttt.env.O else 0 for c in env.board)
                assert abs(numx - numo) <= 1
                assert numx <= 5
                assert numo <= 5
                assert 9 - numx - numo == num_empty - 1
                assert i < 10
                i += 1
                if not done:
                    assert reward == 0
            s = ttt.env.status(env.board)
            assert s != 0
            if s == ttt.env.O: assert reward == -1
            elif s == ttt.env.X: assert reward == 1
            else: assert reward == 0
