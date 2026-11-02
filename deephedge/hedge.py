"""Policy-driven hedging simulation (autograd-differentiable).

We are short one European call. The engine walks each price path, asks the
``policy`` for a share holding at every rebalancing step given only information
available at that step, and accrues cash self-financingly with proportional
transaction costs. It returns the terminal P&L per path.

Written in ``autograd.numpy`` so that when ``policy`` is a neural network the
loss is differentiable w.r.t. its parameters; prices are constants.
"""

from __future__ import annotations

from typing import Callable

import autograd.numpy as anp
import numpy as np

from .costs import proportional_cost
from .gbm import bs_call_price

Policy = Callable[[anp.ndarray], anp.ndarray]  # features (n_paths, 3) -> holding (n_paths,)


def simulate_hedged_pnl(
    paths: np.ndarray,
    policy: Policy,
    K: float,
    r: float,
    sigma: float,
    T: float,
    cost_rate: float,
    premium: float | None = None,
) -> anp.ndarray:
    """Terminal P&L per path of hedging a short European call.

    ``policy`` receives features ``[time_to_maturity, log(S/K), current_holding]``
    of shape (n_paths, 3) and returns the target holding (n_paths,). The holding
    at step k uses only data at step k, so the strategy is causal by construction.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    if premium is None:
        premium = float(bs_call_price(paths[0, 0], K, r, sigma, T))

    cash = anp.ones(n_paths) * premium
    shares = anp.zeros(n_paths)
    for k in range(n_steps):
        s_k = paths[:, k]
        ttm = anp.ones(n_paths) * (T - k * dt)
        logm = anp.log(s_k / K)
        feats = anp.stack([ttm, logm, shares], axis=1)
        holding = policy(feats)
        trade = holding - shares
        cash = cash - trade * s_k - proportional_cost(anp.abs(trade) * s_k, cost_rate)
        shares = holding
        cash = cash * anp.exp(r * dt)

    s_T = paths[:, -1]
    payoff = anp.maximum(s_T - K, 0.0)
    cash = cash + shares * s_T - proportional_cost(anp.abs(shares) * s_T, cost_rate) - payoff
    return cash
