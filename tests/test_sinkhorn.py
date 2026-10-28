"""Tests for the classical Sinkhorn algorithm."""

import numpy as np
import ot
import pytest

from lorentz_sinkhorn.sinkhorn import sinkhorn


def make_problem(p, n=5, m=6, seed=0, uniform=True):
    """Random point clouds in the unit square with cost |x - y|^p."""
    # Fixed seed, so every test run sees the same random points
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 2))
    y = rng.uniform(size=(m, 2))
    if uniform:
        a = np.full(n, 1 / n)
        b = np.full(m, 1 / m)
    else:
        # Random positive weights summing to 1
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
    # Cost matrix C[i, j] = |x[i] - y[j]|^p, the cost behind W_p
    C = ot.dist(x, y, metric="euclidean") ** p
    return a, b, C


# Each case: (exponent p, uniform weights?)
#   p = 1: cost of W_1
#   p = 2: cost of W_2
#   p = 8: large p, a proxy for W_infinity
CASES = [
    (1, True),
    (2, True),
    (2, False),
    (8, True),
]


# Run this test once for every entry of CASES
@pytest.mark.parametrize("p, uniform", CASES)
def test_marginals(p, uniform):
    """The computed coupling must have row sums a and column sums b."""
    a, b, C = make_problem(p, uniform=uniform)
    P, _ = sinkhorn(a, b, C, eps=0.1)
    assert np.allclose(P.sum(axis=1), a, atol=1e-8)
    assert np.allclose(P.sum(axis=0), b, atol=1e-8)


@pytest.mark.parametrize("p, uniform", CASES)
def test_matches_pot(p, uniform):
    """The computed coupling must agree with the reference solver from POT."""
    a, b, C = make_problem(p, uniform=uniform)
    P, _ = sinkhorn(a, b, C, eps=0.1)
    # Run POT with a much stricter tolerance, so its answer is essentially exact
    P_ref = ot.sinkhorn(a, b, C, reg=0.1, stopThr=1e-12, numItermax=100_000)
    assert np.allclose(P, P_ref, atol=1e-8)