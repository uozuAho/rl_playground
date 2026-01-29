import random

from utils.maths import softmax, is_prob_dist, add_dirichlet_noise


def test_softmax():
    for _ in range(10):
        x = [random.randint(0, 10) for _ in range(random.randint(1, 10))]
        s = softmax(x)
        assert is_prob_dist(s)


def test_add_dirichlet_noise():
    for _ in range(10):
        x = [random.randint(0, 10) for _ in range(random.randint(1, 10))]
        s = softmax(x)
        d = add_dirichlet_noise(s, random.uniform(0.01, 2.0), random.uniform(0.01, 1.0))
        assert is_prob_dist(d)
