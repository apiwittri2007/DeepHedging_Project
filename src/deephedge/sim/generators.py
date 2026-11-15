"""Training-data generators that mimic how real windows are built.

Every generator returns daily *log-return* paths of length 84: 63 warm-up days
from which the trader estimates volatility exactly as on real data, then the
21-day hedging window. :func:`windows_from_returns` turns them into the same
(path, trailing vol) pairs that :func:`deephedge.data.realdata.build_windows`
produces from real prices, so every training arm faces the same
volatility-estimation error the real test set has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln

TRADING_DAYS = 252
VOL_LOOKBACK = 63
HORIZON = 21
N_DAYS = VOL_LOOKBACK + HORIZON  # 84
MIN_GARCH_OBS = 500


def windows_from_returns(lr: np.ndarray, vol_lookback: int = VOL_LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """(n, 84) log returns -> hedging paths (n, 22) starting at 100 and trailing vol (n,)."""
    lr = np.atleast_2d(lr)
    prices = np.exp(np.concatenate([np.zeros((len(lr), 1)), np.cumsum(lr, axis=1)], axis=1))
    window = prices[:, vol_lookback:]
    paths = window / window[:, :1] * 100.0
    trailing = lr[:, :vol_lookback].std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    return paths, trailing


def gbm_returns(sigmas: np.ndarray, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-drift GBM daily log returns, one annual vol per path."""
    sig = np.asarray(sigmas, dtype=float)[:, None]
    dt = 1.0 / TRADING_DAYS
    return -0.5 * sig**2 * dt + sig * np.sqrt(dt) * rng.standard_normal((len(sig), n_days))


def stationary_bootstrap(r: np.ndarray, n: int, n_days: int, mean_block: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length (mean ``mean_block``).

    Resampling real returns in blocks keeps their fat tails and short-range
    volatility clustering without assuming a model.
    """
    r = np.asarray(r, dtype=float)
    m = len(r)
    idx = np.empty((n, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, m, n)
    p_new = 1.0 / mean_block
    for t in range(1, n_days):
        new = rng.random(n) < p_new
        idx[:, t] = np.where(new, rng.integers(0, m, n), (idx[:, t - 1] + 1) % m)
    return r[idx]


@dataclass(frozen=True)
class GarchT:
    """GARCH(1,1) with variance-1 Student-t shocks: r = mu + sqrt(h) z."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float


def garch_conditional_variance(model: GarchT, r: np.ndarray) -> np.ndarray:
    """Fitted conditional variance h_t for a return series (h_0 = sample variance)."""
    e2 = (np.asarray(r) - model.mu) ** 2
    h0 = np.var(r)
    x = model.omega + model.alpha * np.r_[h0, e2[:-1]]
    h, _ = lfilter([1.0], [1.0, -model.beta], x, zi=[model.beta * h0])
    return np.maximum(h, 1e-12)


def _unpack(theta: np.ndarray, nu_min: float = 2.05) -> GarchT:
    mu, a, b, c, d = np.clip(theta, -50, 50)
    persistence = 0.9999 / (1 + np.exp(-b))
    share = 1 / (1 + np.exp(-c))
    return GarchT(mu=mu, omega=np.exp(a), alpha=persistence * share,
                  beta=persistence * (1 - share), nu=nu_min + np.exp(d))


def _neg_loglik(theta: np.ndarray, r: np.ndarray, nu_min: float = 2.05) -> float:
    m = _unpack(theta, nu_min)
    h = garch_conditional_variance(m, r)
    e2 = (r - m.mu) ** 2
    nu = m.nu
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e2 / (h * (nu - 2))))
    return -float(ll.sum())


def fit_garch_t(r: np.ndarray, nu_min: float = 2.05) -> GarchT:
    """Maximum-likelihood GARCH(1,1)-t fit (needs at least 500 returns).

    ``nu_min`` is the lowest allowed Student-t degrees of freedom; 4 keeps the
    shock kurtosis finite (used by the A2b sensitivity arm).
    """
    r = np.asarray(r, dtype=float)
    if len(r) < MIN_GARCH_OBS:
        raise ValueError(f"GARCH fit needs at least {MIN_GARCH_OBS} returns, got {len(r)}")
    var = r.var()
    p0, share0 = 0.95, 0.1 / 0.95
    theta0 = np.array([r.mean(), np.log(var * (1 - p0)), np.log(p0 / (0.9999 - p0)),
                       np.log(share0 / (1 - share0)), np.log(max(8.0 - nu_min, 0.5))])
    res = minimize(_neg_loglik, theta0, args=(r, nu_min), method="L-BFGS-B")
    return _unpack(res.x, nu_min)


def simulate_garch_t(model: GarchT, n: int, n_days: int, rng: np.random.Generator,
                     h0=None) -> np.ndarray:
    """Simulate (n, n_days) returns; ``h0`` = starting variance (scalar or per path)."""
    uncond = model.omega / max(1e-6, 1 - model.alpha - model.beta)
    h = np.broadcast_to(np.asarray(uncond if h0 is None else h0, dtype=float), (n,)).copy()
    scale = np.sqrt((model.nu - 2) / model.nu)
    out = np.empty((n, n_days))
    for t in range(n_days):
        e = np.sqrt(h) * rng.standard_t(model.nu, n) * scale
        out[:, t] = model.mu + e
        h = model.omega + model.alpha * e**2 + model.beta * h
    return out


def sample_mixture(per_stock: dict, n: int, draw: Callable, rng: np.random.Generator,
                   weights: dict | None = None) -> np.ndarray:
    """Draw ``n`` paths across stocks, each stock's share ∝ its weight (default: data length).

    ``draw(item, k, rng)`` must return a (k, n_days) array for one stock's item.
    """
    names = list(per_stock)
    w = np.array([(weights or {}).get(k, len(per_stock[k]) if weights is None else 0) for k in names],
                 dtype=float)
    counts = rng.multinomial(n, w / w.sum())
    parts = [draw(per_stock[k], c, rng) for k, c in zip(names, counts) if c > 0]
    out = np.concatenate(parts, axis=0)
    return out[rng.permutation(len(out))]
