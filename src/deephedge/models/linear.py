"""Linear-regression hedging benchmarks (Ruf & Wang 2022; Hull & White 2017).

The holding is BS delta plus a quadratic correction in delta,
``h = delta + c0 + c1*delta + c2*delta**2``, with the three coefficients fitted
by least squares to minimise the variance of zero-cost hedging P&L on real
calibration windows.

``form="HW"`` is the Hull & White (2017) minimum-variance delta,
``h = delta + n(d1) * (a + b*delta + c*delta**2)`` (their vega / (S sqrt(T))
scaling equals n(d1)); with real option prices unavailable, its coefficients are
fitted the same way, to the variance of hedging P&L on real windows. With zero rates the P&L is linear in the coefficients,
``pnl = pnl_delta + sum_j c_j X_j`` with ``X_j = sum_k f_kj dS_k`` and
``f = [1, delta, delta**2]``, so the fit is one closed-form regression.
"""

from __future__ import annotations

import numpy as np

from deephedge.hedging.delta import feature_delta_policy
from deephedge.hedging.hedge import simulate_hedged_pnl
from scipy.stats import norm

from deephedge.sim.gbm import bs_call_delta, path_sigma

FORMS = ("L", "HW")


def _basis(d, form: str):
    """Regressors for the correction term, from BS delta (r = 0, so d1 = Phi^-1(delta))."""
    b = np.stack([np.ones_like(d), d, d * d], axis=-1)
    if form == "HW":
        b = b * norm.pdf(norm.ppf(np.clip(d, 1e-9, 1 - 1e-9)))[..., None]
    elif form != "L":
        raise ValueError(f"form must be one of {FORMS}")
    return b


def _regressors(paths: np.ndarray, sigma, K: float, T: float, form: str = "L") -> np.ndarray:
    n, cols = paths.shape
    steps = cols - 1
    dt = T / steps
    sigma = path_sigma(sigma, n)
    X = np.zeros((n, 3))
    for k in range(steps):
        d = bs_call_delta(paths[:, k], K, 0.0, sigma, max(T - k * dt, 1e-8))
        dS = paths[:, k + 1] - paths[:, k]
        X += _basis(d, form) * dS[:, None]
    return X


def fit_linear_correction(paths: np.ndarray, sigma, K: float = 100.0, T: float = 21 / 252,
                          form: str = "L") -> np.ndarray:
    """Coefficients (3,) minimising Var(zero-cost P&L) of the corrected delta hedge."""
    base = np.asarray(simulate_hedged_pnl(paths, feature_delta_policy(), K, 0.0, sigma, T, 0.0))
    X = _regressors(paths, sigma, K, T, form)
    Xc, yc = X - X.mean(0), base - base.mean()
    coef, *_ = np.linalg.lstsq(Xc, -yc, rcond=None)
    return coef


def linear_policy(coef, form: str = "L"):
    c = np.asarray(coef, dtype=float)

    def policy(feats):
        d = np.asarray(feats[:, 3], dtype=float)
        return d + _basis(d, form) @ c

    return policy
