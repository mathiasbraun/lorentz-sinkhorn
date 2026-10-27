# Lorentz-Sinkhorn

Entropic optimal transport under a causality constraint, solved with the Sinkhorn algorithm.
Mass may only move along a causal relation (on Minkowski space or a directed acyclic graph),
and the cost is a power of the time separation, which is maximized.

Companion code to the upcoming paper and the talk *Entropic Lorentz optimal transport and the Sinkhorn algorithm*.

**Status:** work in progress.

## Problem

Given probability vectors $a \in \mathbb{R}^n$, $b \in \mathbb{R}^m$ and a cost matrix $C$,
entropic optimal transport seeks the coupling $P$ with marginals $a$ and $b$ minimizing

$$\langle C, P\rangle - \varepsilon H(P), \qquad H(P) = -\sum_{i,j} P_{ij} (\log P_{ij} - 1).$$

The Sinkhorn algorithm finds it by alternately rescaling the rows and columns of the
Gibbs kernel $K = e^{-C/\varepsilon}$. The Lorentz version (in progress) maximizes a
time-separation cost and forbids transport between non-causally related points.

## Usage

```python
import numpy as np
from lorentz_sinkhorn.sinkhorn import sinkhorn

a = np.array([0.5, 0.5])
b = np.array([0.5, 0.5])
C = np.array([[0.0, 1.0], [1.0, 0.0]])

P, n_iter = sinkhorn(a, b, C, eps=0.1)
```

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1      # on Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Tests

    pytest