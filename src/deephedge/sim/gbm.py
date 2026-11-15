"""Geometric Brownian Motion paths and Black-Scholes call formulas.

Path simulation and the closed-form benchmark use plain NumPy: in training the
simulated prices are constants (data), and only the network parameters are
differentiated, so these need not be autograd-traced.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def path_sigma(sigma, n_paths: int):
    """Return ``sigma`` as a scalar or a validated (n_paths,) array (per-path vol)."""
    if np.ndim(sigma) == 0:
        return float(sigma)
    sig = np.asarray(sigma, dtype=float).reshape(-1)
    if sig.shape[0] != n_paths:
        raise ValueError(f"sigma must be a scalar or have length n_paths={n_paths}, got {sig.shape[0]}")
    return sig


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM price paths; returns array of shape (n_paths, n_steps + 1).

    ``sigma`` may be a scalar or a per-path array of length ``n_paths``.
    """
    sig = path_sigma(sigma, n_paths)
    sigma = sig[:, None] if np.ndim(sig) else sig
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
