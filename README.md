# Lorentz-Sinkhorn Algorithm

Entropic optimal transport under a causality constraint, solved with the Sinkhorn algorithm.
Mass may only move along a causal relation (on Minkowski space or a directed acyclic graph),
and the cost is a power of the time separation, which is maximized.

Companion code to the upcoming paper and the talk *Entropic Lorentz optimal transport and the Sinkhorn algorithm*.

**Status:** work in progress.

## Setup

    python -m venv .venv
    .venv\Scripts\Activate.ps1      # on Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Tests

    pytest