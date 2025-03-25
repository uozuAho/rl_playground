import random

import ttt.env2 as ttt


def test_fuzz():
    env = ttt.Env()
    for _ in range(100):
        env.reset()
        done = False
        i = 0
        while not done:
            num_empty = sum(1 if c == ttt.EMPTY else 0 for c in env.board)
            va = list(env.valid_actions())
            assert len(va) == num_empty
            for a in va:
                assert env.board[a] == ttt.EMPTY
            action = random.choice(va)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            assert obs == env.board
            numx = sum(1 if c == ttt.X else 0 for c in env.board)
            numo = sum(1 if c == ttt.O else 0 for c in env.board)
            assert abs(numx - numo) <= 1
            assert numx <= 5
            assert numo <= 5
            assert 9 - numx - numo == num_empty - 1
            assert i < 10
            i += 1
            if not done:
                assert reward == 0
