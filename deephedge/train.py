"""Train a deep hedger by minimising the CVaR of terminal hedging P&L.

Uses the Rockafellar-Uryasev objective (CVaR as a minimisation over a VaR level
w), differentiated through the whole simulated hedging path with autograd and
optimised with Adam. Both the network parameters and w are learned jointly.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from autograd import grad
from autograd.misc.flatten import flatten

from .hedge import simulate_hedged_pnl
from .metrics import ru_cvar_objective
from .network import init_params, make_policy


def train_deep_hedger(
    paths: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
    cost_rate: float,
    alpha: float = 0.95,
    layer_sizes: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.01,
    seed: int = 0,
) -> tuple[dict, list[float]]:
    """Return trained params ``{"net", "w"}`` and the per-epoch loss history."""
    layer_sizes = layer_sizes or [3, 16, 16, 1]
    rng = np.random.default_rng(seed)
    theta = {"net": init_params(layer_sizes, seed=seed), "w": 0.0}
    flat, unflatten = flatten(theta)

    def loss_from_flat(flat_vec, batch):
        th = unflatten(flat_vec)
        pnl = simulate_hedged_pnl(batch, make_policy(th["net"]), K, r, sigma, T, cost_rate)
        return ru_cvar_objective(pnl, th["w"], alpha)

    gloss = grad(loss_from_flat)
    m = np.zeros_like(flat)
    v = np.zeros_like(flat)
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = paths.shape[0]
    history: list[float] = []

    for t in range(1, epochs + 1):
        batch = paths[rng.integers(0, n, size=min(batch_size, n))]
        loss = loss_from_flat(flat, batch)
        g = gloss(flat, batch)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        flat = flat - lr * mh / (np.sqrt(vh) + eps)
        history.append(float(loss))

    return unflatten(flat), history
