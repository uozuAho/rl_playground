import pytest

from utils.qtable import StateValueTable, greedy_probs
import ttt.env as t3

t = StateValueTable.from_dict(
    {
        "...|...|...": 0.0,
        "...|.x.|...": 1.0,
        "...|.xo|...": -1.0,
    }
)


# Parameterized test
@pytest.mark.parametrize(
    "board, expected",
    [
        ("...|...|...", [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("...|.x.|...", [0, 0, 0, 0, 0, 1, 0, 0, 0]),
    ],
)
def test_greedy_probs(board, expected):
    env = t3.TttEnv.from_str(board)
    probs = greedy_probs(t, env)
    assert probs == expected
