"""Linear-regression hedging benchmark (in the spirit of Ruf & Wang 2022).

The holding is BS delta plus a quadratic correction in delta,
``h = delta + c0 + c1*delta + c2*delta**2``, with the three coefficients fitted
by least squares to minimise the variance of zero-cost hedging P&L on real
calibration windows. With zero rates the P&L is linear in the coefficients,
``pnl = pnl_delta + sum_j c_j X_j`` with ``X_j = sum_k f_kj dS_k`` and
``f = [1, delta, delta**2]``, so the fit is one closed-form regression.
"""

from __future__ import annotations

import numpy as np

from deephedge.hedging.delta import feature_delta_policy
from deephedge.hedging.hedge import simulate_hedged_pnl
from deephedge.sim.gbm import bs_call_delta, path_sigma


def _regressors(paths: np.ndarray, sigma, K: float, T: float) -> np.ndarray:
    n, cols = paths.shape
    steps = cols - 1
    dt = T / steps
    sigma = path_sigma(sigma, n)
    X = np.zeros((n, 3))
    for k in range(steps):
        d = bs_call_delta(paths[:, k], K, 0.0, sigma, max(T - k * dt, 1e-8))
        dS = paths[:, k + 1] - paths[:, k]
        X += np.stack([np.ones(n), d, d * d], axis=1) * dS[:, None]
    return X


def fit_linear_correction(paths: np.ndarray, sigma, K: float = 100.0, T: float = 21 / 252) -> np.ndarray:
    """Coefficients (3,) minimising Var(zero-cost P&L) of the corrected delta hedge."""
    base = np.asarray(simulate_hedged_pnl(paths, feature_delta_policy(), K, 0.0, sigma, T, 0.0))
    X = _regressors(paths, sigma, K, T)
    Xc, yc = X - X.mean(0), base - base.mean()
    coef, *_ = np.linalg.lstsq(Xc, -yc, rcond=None)
    return coef


def linear_policy(coef):
    c0, c1, c2 = (float(c) for c in coef)

    def policy(feats):
        d = feats[:, 3]
        return d + c0 + c1 * d + c2 * d * d

    return policy
