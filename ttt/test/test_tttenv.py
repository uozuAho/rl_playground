import pytest

import ttt.env
from ttt.env import TicTacToeEnv


def test_random_moves_to_end():
    env = TicTacToeEnv()
    env.reset()

    terminated = False
    i = 0
    while not terminated:
        action = env.action_space.sample(mask=env.valid_action_mask())
        obs, reward, terminated, truncated, info = env.step(action)
        assert i < 10

    assert env.is_game_over


def test_valid_actions():
    env = TicTacToeEnv()
    env.reset()
    assert list(env.valid_actions()) == list(range(9))
    for i in range(7):
        env.step(i)
        env.render()
        assert i not in list(env.valid_actions())
        # x wins by 7


def test_throw_on_invalid_action():
    env = TicTacToeEnv()
    env.reset()
    env.step(0)
    with pytest.raises(Exception):
        env.step(0)


def test_game_over_on_invalid_action():
    env = TicTacToeEnv(on_invalid_action=ttt.env.INVALID_ACTION_GAME_OVER)
    env.reset()
    env.step(0)
    obs, reward, terminated, _, _ = env.step(0)
    assert reward < 0
    assert terminated
    assert env.is_game_over


def test_copy():
    env = TicTacToeEnv()
    env.reset()
    env2 = env.copy()
    env.step(1)
    assert 1 not in env.valid_actions()
    assert 1 in env2.valid_actions()
