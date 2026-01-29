import math

import numpy as np


def softmax(arr: list[float]):
    mx = max(arr)
    ex = [math.exp(x - mx) for x in arr]
    sm = sum(ex)
    return [x / sm for x in ex]


def is_prob_dist(numbers):
    return all((0.99 < sum(numbers) < 1.01, all(0 <= x <= 1.0 for x in numbers)))


def add_dirichlet_noise(p, alpha, epsilon):
    """Params:
    alpha: spikiness/distribution. lower = more spiky, less distributed noise
    epsilon: amount of noise injected"""
    # assert is_prob_dist(p)
    assert 0 <= epsilon <= 1.0
    noise = np.random.dirichlet([alpha] * len(p))
    return (1 - epsilon) * np.asarray(p) + epsilon * noise


def kl_divergence(p, q):
    assert is_prob_dist(p)
    assert is_prob_dist(q)
    p = np.asarray(p) + 1e-32  # ensure nonzero
    q = np.asarray(q) + 1e-32
    return np.sum(p * np.log(p / q))
