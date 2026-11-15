"""A small multilayer perceptron policy with autograd parameters.

Parameters are a list of (W, b) tuples so autograd can differentiate the
hedging loss with respect to the whole network. Input features are
[time_to_maturity, log-moneyness, current_holding]; the scalar output is the
target share holding.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np


def init_params(layer_sizes: list[int], seed: int | None = None) -> list[tuple]:
    """He/Xavier-ish initialisation for each layer; returns list of (W, b)."""
    rng = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_out))
        b = np.zeros(n_out)
        params.append((W, b))
    return params


def mlp_forward(params: list[tuple], feats) -> anp.ndarray:
    """Forward pass; tanh hidden activations, linear output. Returns (n_paths,)."""
    a = feats
    for W, b in params[:-1]:
        a = anp.tanh(anp.dot(a, W) + b)
    W, b = params[-1]
    out = anp.dot(a, W) + b
    return out[:, 0]


def make_policy(params: list[tuple]):
    """Wrap params into a hedging policy (features -> holding)."""
    return lambda feats: mlp_forward(params, feats)
