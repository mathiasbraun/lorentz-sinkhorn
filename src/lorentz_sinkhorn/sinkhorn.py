"""Classical entropic optimal transport via the Sinkhorn algorithm."""

import numpy as np


def sinkhorn(a, b, C, eps, max_iter=10_000, tol=1e-9):
    """Solve entropic optimal transport with the Sinkhorn algorithm.

    Minimizes  <C, P> - eps * H(P)  over couplings P with marginals a and b.

    Parameters
    ----------
    a : array of shape (n,)
        Source weights (nonnegative, sum 1).
    b : array of shape (m,)
        Target weights (nonnegative, sum 1).
    C : array of shape (n, m)
        Cost matrix.
    eps : float
        Regularization strength (> 0).
    max_iter : int
        Maximal number of iterations.
    tol : float
        Stop once the row marginal error is below tol.

    Returns
    -------
    P : array of shape (n, m)
        The entropic optimal coupling.
    n_iter : int
        Number of iterations performed.
    """
    K = np.exp(-C / eps)          # Gibbs kernel
    u = np.ones_like(a)
    v = np.ones_like(b)

    for n_iter in range(1, max_iter + 1):
        u = a / (K @ v)           # match row sums to a
        v = b / (K.T @ u)         # match column sums to b

        P = u[:, None] * K * v[None, :]
        err = np.abs(P.sum(axis=1) - a).sum()
        if err < tol:
            break

    return P, n_iter