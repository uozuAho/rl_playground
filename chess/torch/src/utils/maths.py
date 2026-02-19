import numpy as np


def add_dirichlet_noise(p, alpha, epsilon):
    """Params:
    alpha: spikiness/distribution. lower = more spiky, less distributed noise
    epsilon: amount of noise injected"""
    assert 0 <= epsilon <= 1.0
    noise = np.random.dirichlet([alpha] * len(p))
    return (1 - epsilon) * np.asarray(p) + epsilon * noise
