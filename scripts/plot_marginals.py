"""Plot classical entropic optimal transport plans for several values of eps."""

import matplotlib.pyplot as plt
import numpy as np

from lorentz_sinkhorn.sinkhorn import sinkhorn

# Two measures on the real line, each with 100 points
n, m = 100, 100
# Generate points in [0, 1] for the source and target measures
x = np.linspace(0.0, 1.0, n)
y = np.linspace(0.0, 1.0, m)

# Source: uniform on [0, 1], target: narrow bump at 0.5 plus a small bump at 0.9.
# The optimal map squeezes the whole interval into the bumps, so its graph is
# an S-shaped curve with a step where it switches to the second bump.
a = np.ones(n)
b = np.exp(-((y - 0.5) ** 2) / 0.005) + 0.3 * np.exp(-((y - 0.9) ** 2) / 0.001)
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

fig.suptitle("Entropic optimal transport plans for the quadratic cost, uniform source")
fig.tight_layout()
fig.savefig("figures/marginals_sinkhorn.png", dpi=150)
print("Saved figures/marginals_sinkhorn.png")