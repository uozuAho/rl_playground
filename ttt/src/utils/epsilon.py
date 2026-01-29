import math


def exp_decay(eps_start: float, eps_end: float, n_total: int, n_current: int) -> float:
    """Return a nicely decaying epsilon value"""
    return eps_end + (eps_start - eps_end) * math.exp(-6.0 * n_current / n_total)


def exp_decay_gen(eps_start: float, eps_end: float, n_total: int):
    for i in range(n_total):
        yield exp_decay(eps_start, eps_end, n_total, i)
