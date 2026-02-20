import numpy as np


def add_dirichlet_noise(p, alpha, epsilon):
    """Params:
    alpha: spikiness/distribution. lower = more spiky, less distributed noise
    epsilon: amount of noise injected"""
    assert 0 <= epsilon <= 1.0
    noise = np.random.dirichlet([alpha] * len(p))
    return (1 - epsilon) * np.asarray(p) + epsilon * noise


def is_prob_dist(numbers):
    return all((0.99 < sum(numbers) < 1.01, all(0 <= x <= 1.0 for x in numbers)))


def heat(p: np.ndarray, temp: float):
    assert is_prob_dist(p)
    assert temp >= 0.0
    pp = p ** (1 / temp)
    pp /= np.sum(pp)
    assert is_prob_dist(pp)
    return pp
