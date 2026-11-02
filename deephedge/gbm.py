"""Geometric Brownian Motion paths and Black-Scholes call formulas.

Path simulation and the closed-form benchmark use plain NumPy: in training the
simulated prices are constants (data), and only the network parameters are
differentiated, so these need not be autograd-traced.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths; returns array of shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths)


def _d1_d2(s, K, r, sigma, tau):
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(s / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    return d1, d1 - sigma * sqrt_tau


def bs_call_price(s, K, r, sigma, tau):
    """Black-Scholes European call price; at tau=0 returns the payoff max(s-K, 0)."""
    s = np.asarray(s, dtype=float)
    if np.ndim(tau) == 0 and tau <= 0:
        return np.maximum(s - K, 0.0)
    d1, d2 = _d1_d2(s, K, r, sigma, tau)
    return s * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(s, K, r, sigma, tau):
    """Black-Scholes call delta; at tau=0 returns the indicator 1{s > K}."""
    s = np.asarray(s, dtype=float)
    if np.ndim(tau) == 0 and tau <= 0:
        return (s > K).astype(float)
    d1, _ = _d1_d2(s, K, r, sigma, tau)
    return norm.cdf(d1)
