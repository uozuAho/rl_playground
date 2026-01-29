import json
import typing as t
import ttt.env as t3
from utils.maths import is_prob_dist


class StateValueTable:
    def __init__(self):
        self._table = {}

    def size(self):
        return len(self._table)

    def value(self, env: t3.TttEnv):
        return self._table.get(self._env2str(env), 0.0)

    def set_value(self, env: t3.TttEnv, value: float):
        self._table[self._env2str(env)] = value

    def values(self) -> t.Iterable[tuple[t3.TttEnv, float]]:
        for bstr, val in self._table.items():
            yield t3.TttEnv.from_str(bstr), val

    def save(self, path):
        with open(path, "w") as ofile:
            ofile.write(json.dumps(self._table, indent=2))

    @staticmethod
    def load(path, str2env=None):
        with open(path, "r") as infile:
            d = json.load(infile)
            return StateValueTable.from_dict(d, str2env)

    @staticmethod
    def from_dict(d, str2env=None):
        if str2env is None:
            str2env = StateValueTable._str2env
        table = StateValueTable()
        for bstr, v in d.items():
            env = str2env(bstr)
            table.set_value(env, v)
        return table

    def _env2str(self, env: t3.TttEnv):
        return env.str1d()

    @staticmethod
    def _str2env(s: str):
        return t3.TttEnv.from_str(s)


def greedy_probs(qtable: StateValueTable, env: t3.TttEnv):
    """NOTE: this assumes the values are [-1,1] for player X"""
    legal_actions = list(env.valid_actions())
    assert legal_actions  # game is over if there's no legal actions
    next_states = []
    for a in legal_actions:
        e = env.copy()
        e.step(a)
        next_states.append(e)
    next_values = [qtable.value(e) for e in next_states]
    if env.current_player == t3.O:
        # invert value if playing O. Assumes value is for X
        next_values = [-x for x in next_values]
    policy = [0] * 9
    for i, a in enumerate(legal_actions):
        policy[a] = next_values[i] - min(next_values)
    # normalise to 0-1
    psum = sum(policy)
    if psum == 0:
        policy = [1 / len(legal_actions) if a in legal_actions else 0 for a in range(9)]
    else:
        policy = [p / psum for p in policy]
    assert is_prob_dist(policy)
    return policy
