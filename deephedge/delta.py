"""Black-Scholes delta hedging policies (benchmarks for deep hedging)."""

from __future__ import annotations

import autograd.numpy as anp

from .gbm import bs_call_delta

_TTM_FLOOR = 1e-8


def _delta_from_features(feats, K, r, sigma):
    ttm = anp.maximum(feats[:, 0], _TTM_FLOOR)
    spot = K * anp.exp(feats[:, 1])
    return bs_call_delta(spot, K, r, sigma, ttm)


def delta_policy(K: float, r: float, sigma: float):
    """Policy that holds the Black-Scholes delta at every step."""
    def policy(feats):
        return _delta_from_features(feats, K, r, sigma)
    return policy


def banded_delta_policy(K: float, r: float, sigma: float, band: float):
    """Delta policy with a no-trade band: hold unless |delta - current| > band."""
    def policy(feats):
        target = _delta_from_features(feats, K, r, sigma)
        current = feats[:, 2]
        trade_needed = anp.abs(target - current) > band
        return anp.where(trade_needed, target, current)
    return policy
