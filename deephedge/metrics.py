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
