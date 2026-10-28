"""Plot classical entropic optimal transport plans for several values of eps."""

import matplotlib.pyplot as plt
import numpy as np

from lorentz_sinkhorn.sinkhorn import sinkhorn

# Two measures on the real line, each with 100 points
n, m = 100, 100
# Generate points in [0, 1] for the source and target measures
x = np.linspace(0.0, 1.0, n)
y = np.linspace(0.0, 1.0, m)

# Source: one bump at 0.3, target: two bumps at 0.2 and 0.8 with equal mass.
# The source must split its mass between the two targets, so the optimal map
# has a jump. The source sits closer to the left target, so the split is
# asymmetric. Widths keep all bumps well inside [0, 1] and resolved by the grid.
a = np.exp(-((x - 0.3) ** 2) / 0.01)
b = np.exp(-((y - 0.2) ** 2) / 0.005) + np.exp(-((y - 0.8) ** 2) / 0.005)
# Normalize the weights to sum to 1
a /= a.sum()
b /= b.sum()

C = (x[:, None] - y[None, :]) ** 2   # Quadratic cost |x - y|^2

eps_values = [0.1, 0.01, 0.001]
fig, axes = plt.subplots(1, len(eps_values), figsize=(12, 4))

for ax, eps in zip(axes, eps_values):
    P, n_iter = sinkhorn(a, b, C, eps)
    ax.imshow(P, origin="lower", extent=[0, 1, 0, 1], cmap="Blues")
    ax.set_title(f"eps = {eps}  ({n_iter} iterations)")
    ax.set_xlabel("target y")
    ax.set_ylabel("source x")

fig.suptitle("Entropic optimal transport plans for the quadratic cost")
fig.tight_layout()
fig.savefig("figures/classical_sinkhorn.png", dpi=150)
print("Saved figures/classical_sinkhorn.png")