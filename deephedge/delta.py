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


def feature_delta_policy():
    """BS delta taken from the engine's delta feature (feats[:, 3]).

    Equivalent to :func:`delta_policy` but uses whatever (possibly per-path)
    volatility the engine priced with, e.g. each real window's trailing vol.
    """
    return lambda feats: feats[:, 3]


def feature_banded_policy(band: float):
    """No-trade-band delta using the engine's delta feature."""
    def policy(feats):
        target, current = feats[:, 3], feats[:, 2]
        return anp.where(anp.abs(target - current) > band, target, current)
    return policy
