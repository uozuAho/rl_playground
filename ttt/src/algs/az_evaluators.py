import random
from pathlib import Path

import utils.maths as maths
import ttt.env as t3
from utils import qtable
import typing as t

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINED_MODELS_PATH = PROJECT_ROOT / "trained_models"


type Pdist = list[float]
type Policy = Pdist
type Value = float
type PV = tuple[Policy, Value]
type PolicyFn = t.Callable[[t3.TttEnv], Policy]
type ValueFn = t.Callable[[t3.TttEnv], Value]
type PvFn = t.Callable[[t3.TttEnv], tuple[Policy, Value]]


def uniform(env) -> PV:
    return [0.1111111] * 9, 0.1


def uniform_batch(envs) -> list[PV]:
    return [([0.1111111] * 9, 0.1) for _ in range(len(envs))]


def random_eval(env) -> PV:
    probs = maths.add_dirichlet_noise([1] * 9, 0.1, 1.0)
    val = random.uniform(-1, 1)
    return probs, val


def winning_move_evaluator(env: t3.TttEnv) -> PV:
    policy = [0.01] * 9

    for action in env.valid_actions():
        test_env = env.copy()
        test_env.step(action)
        if test_env.status() == env.current_player:
            policy[action] = 0.9

    total = sum(policy)
    policy = [p / total for p in policy]
    value = 0.0
    return policy, value


def make_greedy_tab_pv_eval() -> PvFn:
    """Use pre-trained tabular state-val table to return greedy next move policy + state val"""
    table = qtable.StateValueTable.load(TRAINED_MODELS_PATH / "tmcts_sym_100k_30")
    return from_pv(to_tab_greedy_p(table), to_tab_v(table))


def make_uniform_tab_v_eval() -> PvFn:
    """Use pre-trained tabular state-val table to return uniform policy + state val"""
    table = qtable.StateValueTable.load(TRAINED_MODELS_PATH / "tmcts_sym_100k_30")
    return from_pv(lambda _: [0.11111] * 9, to_tab_v(table))


def to_tab_greedy_p(table: qtable.StateValueTable) -> PolicyFn:
    def eval(env: t3.TttEnv):
        probs = qtable.greedy_probs(table, env)
        return probs

    return eval


def to_tab_v(table: qtable.StateValueTable) -> ValueFn:
    def eval(env: t3.TttEnv):
        val = table.value(env)
        return val if env.current_player == t3.X else -val

    return eval


def from_pv(pfunc: PolicyFn, vfunc: ValueFn) -> PvFn:
    def eval(env: t3.TttEnv):
        return pfunc(env), vfunc(env)

    return eval


def _add_v_noise(v: float, amount: float):
    assert 0 <= amount <= 1.0
    rand_v = random.uniform(-amount, amount)
    nv = (1 - amount) * v + amount * rand_v
    assert -1.0 <= nv <= 1.0
    return nv


def make_noisy_pv(pv: PvFn, pa: float, va: float) -> PvFn:
    """Add noise to a given pv function.
    Params:
    pa: policy noise amount: 0 = no noise, 1 = practically random
    va: value noise amount
    """

    def noisy_pv(env):
        p, v = pv(env)
        p = maths.add_dirichlet_noise(p, 0.1, pa)
        v = _add_v_noise(v, va)
        assert maths.is_prob_dist(p)
        return p, v

    return noisy_pv
