"""Tests for the classical Sinkhorn algorithm."""

import numpy as np
import ot

from lorentz_sinkhorn.sinkhorn import sinkhorn


def make_problem(n=5, m=6, seed=0):
    """Random point clouds in the plane with uniform weights."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=(m, 2))
    a = np.full(n, 1 / n)
    b = np.full(m, 1 / m)
    C = ot.dist(x, y)            # Squared Euclidean distances
    return a, b, C


def test_marginals():
    a, b, C = make_problem()
    P, _ = sinkhorn(a, b, C, eps=0.5)
    assert np.allclose(P.sum(axis=1), a, atol=1e-8)
    assert np.allclose(P.sum(axis=0), b, atol=1e-8)


def test_matches_pot():
    a, b, C = make_problem()
    P, _ = sinkhorn(a, b, C, eps=0.5)
    P_ref = ot.sinkhorn(a, b, C, reg=0.5, stopThr=1e-12, numItermax=100_000)
    assert np.allclose(P, P_ref, atol=1e-8)