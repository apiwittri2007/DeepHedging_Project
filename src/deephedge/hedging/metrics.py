"""Risk metrics for hedging P&L (loss = -P&L)."""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np


def cvar(pnl, alpha: float = 0.95) -> float:
    """Conditional Value-at-Risk of P&L at level ``alpha`` (a positive loss number).

    The mean of the worst ``1 - alpha`` fraction of outcomes, sign-flipped so a
    larger number means a worse tail. ``cvar=5`` means the average P&L in the
    bad tail is -5.
    """
    pnl = np.sort(np.asarray(pnl))
    n = len(pnl)
    k = max(1, int(np.ceil((1.0 - alpha) * n)))
    return float(-pnl[:k].mean())


def ru_cvar_objective(pnl, w, alpha: float = 0.95):
    """Rockafellar-Uryasev CVaR objective (differentiable; minimised over w gives CVaR).

    CVaR_alpha(loss) = min_w [ w + 1/(1-alpha) * E[max(loss - w, 0)] ], loss = -pnl.
    """
    loss = -pnl
    return w + anp.mean(anp.maximum(loss - w, 0.0)) / (1.0 - alpha)


def summary(pnl) -> dict:
    """mean, std, 95% CVaR, 95% VaR and worst outcome of a P&L sample."""
    pnl = np.asarray(pnl)
    return {
        "mean": float(pnl.mean()),
        "std": float(pnl.std()),
        "cvar95": cvar(pnl, 0.95),
        "var95": float(-np.quantile(pnl, 0.05)),
        "min": float(pnl.min()),
    }


def paired_bootstrap_cvar_diff(pnl_a, pnl_b, alpha: float = 0.95, n_boot: int = 2000, seed: int = 0,
                               clusters=None) -> dict:
    """Is strategy A's CVaR really lower than B's? Paired bootstrap over windows.

    Both strategies were run on the same windows, so we resample window indices
    jointly. Returns ``diff = CVaR(A) - CVaR(B)`` (negative = A has less tail
    risk), a 95% percentile CI, and the share of resamples where A is not better.

    ``clusters`` (optional, one label per window, e.g. the start month) resamples
    whole clusters instead of single windows, so outcomes that share a common
    shock are not treated as independent — a more conservative interval.

    Either input may be 2-D (seeds x windows): its CVaR is then the mean of the
    per-seed CVaRs, and resampling draws the same windows for every seed.
    """
    a, b = np.asarray(pnl_a), np.asarray(pnl_b)

    def risk(x, idx=None):
        x = x if idx is None else x[..., idx]
        return float(np.mean([cvar(row, alpha) for row in x])) if x.ndim == 2 else cvar(x, alpha)

    rng = np.random.default_rng(seed)
    n = a.shape[-1]
    groups = None
    if clusters is not None:
        _, inv = np.unique(np.asarray(clusters), return_inverse=True)
        groups = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        diffs[i] = risk(a, idx) - risk(b, idx)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": risk(a) - risk(b), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_not_better": float((diffs >= 0).mean())}
