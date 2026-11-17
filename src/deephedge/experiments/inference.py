"""Joint inference for the training-data study: multiple testing and economic size.

Many arm x cost comparisons are tested against BS delta, so single-test p-values
overstate significance. This module draws one set of month-clustered bootstrap
resamples, shared by every comparison (so their dependence is kept), and reports:

* Holm (1979) step-down adjusted p-values, and
* Romano & Wolf (2005) step-down adjusted p-values (standardised by each test's
  bootstrap standard error), which use the
  joint bootstrap distribution and are less conservative than Holm.

It also converts CVaR differences into money: HK$ per HK$1m of option notional,
and percent of the average option premium.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.95


def cluster_labels(months: np.ndarray, span: int = 1) -> np.ndarray:
    """Group "YYYY-MM" labels into blocks of ``span`` calendar months (1 = month, 3 = quarter).

    A 21-day window starting late in a month runs into the next, so neighbouring
    month clusters share shocks; coarser blocks are a robustness check on that.
    """
    ym = np.array([int(m[:4]) * 12 + int(m[5:7]) - 1 for m in np.asarray(months, dtype=str)])
    return ym // span


def month_counts(months: np.ndarray, n_boot: int, seed: int = 0, span: int = 1) -> np.ndarray:
    """(n_boot, n_windows) integer weights: whole clusters of ``span`` months drawn with replacement."""
    months = cluster_labels(months, span) if span != 1 or np.asarray(months).dtype.kind in "US" else months
    uniq, inv = np.unique(months, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    per_month = np.zeros((n_boot, len(uniq)), dtype=np.int32)
    np.add.at(per_month, (np.arange(n_boot)[:, None], draws), 1)
    return per_month[:, inv]


def weighted_cvar(pnl: np.ndarray, counts: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """CVaR of the resample where window i appears counts[b, i] times (exactly as ``metrics.cvar``)."""
    order = np.argsort(pnl)
    x = pnl[order]
    c = counts[:, order].astype(np.int64)
    k = np.maximum(1, np.ceil((1 - alpha) * c.sum(axis=1)).astype(np.int64))
    before = np.cumsum(c, axis=1) - c
    take = np.clip(k[:, None] - before, 0, c)
    return -(take * x).sum(axis=1) / k


def seed_avg_cvar(pnl2d: np.ndarray, counts: np.ndarray | None = None, alpha: float = ALPHA):
    """Seed-averaged CVaR; with ``counts``, one value per bootstrap resample."""
    pnl2d = np.atleast_2d(pnl2d)
    if counts is None:
        counts = np.ones((1, pnl2d.shape[1]), dtype=np.int32)
        return float(np.mean([weighted_cvar(row, counts, alpha)[0] for row in pnl2d]))
    return np.mean([weighted_cvar(row, counts, alpha) for row in pnl2d], axis=0)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def romano_wolf(t: np.ndarray, t_star: np.ndarray) -> np.ndarray:
    """Step-down adjusted p-values for one-sided tests of H0: diff >= 0 (reject for small t).

    ``t`` (m,) are observed studentised statistics; ``t_star`` (B, m) are the
    bootstrap statistics centred at the observed difference.
    """
    m = len(t)
    order = np.argsort(t)                 # most significant (most negative) first
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        active = order[rank:]
        B = len(t_star)
        p = float((1 + np.sum(t_star[:, active].min(axis=1) <= t[i])) / (B + 1))
        running = max(running, p)
        adj[i] = running
    return adj


def multiple_testing(res: dict, vs: str = "delta", arms=None, costs=None,
                     n_boot: int = 2000, seed: int = 0, span: int = 1) -> list[dict]:
    """Every (arm, cost) vs ``vs``, with raw, Holm and Romano-Wolf p-values (one family).

    ``span`` sets the bootstrap cluster: 1 = calendar month (pre-registered), 2, 3 = quarter.
    p-values use (1 + #)/(B + 1), so they are never exactly zero.
    """
    raw = res["raw"]
    pairs = []
    for key in raw:
        arm, cost = key.split("|")
        if arm == vs or (arms is not None and arm not in arms) or (costs is not None and float(cost) not in costs):
            continue
        if f"{vs}|{cost}" in raw:
            pairs.append((arm, cost))
    if not pairs:
        return []
    counts = month_counts(res["months"], n_boot, seed, span)
    diffs, stars = [], []
    for arm, cost in pairs:
        a, b = raw[f"{arm}|{cost}"], raw[f"{vs}|{cost}"]
        common = a["mask"] & b["mask"]
        pa, pb = a["pnl"][:, common[a["mask"]]], b["pnl"][:, common[b["mask"]]]
        cnt = counts[:, common]
        d = seed_avg_cvar(pa) - seed_avg_cvar(pb)
        d_star = seed_avg_cvar(pa, cnt) - seed_avg_cvar(pb, cnt)
        diffs.append(d)
        stars.append(d_star)
    d = np.array(diffs)
    D = np.stack(stars, axis=1)                         # (B, m)
    se = D.std(axis=0, ddof=1)
    se = np.where(se > 0, se, np.inf)
    t = d / se
    t_star = (D - d) / se
    p_raw = (1 + np.sum(t_star <= t, axis=0)) / (n_boot + 1)
    p_holm, p_rw = holm(p_raw), romano_wolf(t, t_star)
    rows = []
    for j, (arm, cost) in enumerate(pairs):
        lo, hi = np.percentile(D[:, j] - d[j], [97.5, 2.5])
        rows.append({"arm": arm, "cost": float(cost), "vs": vs, "cluster_months": span,
                     "diff": float(d[j]), "se": float(se[j]),
                     "ci_low": float(d[j] - lo), "ci_high": float(d[j] - hi),
                     "p_raw": float(p_raw[j]), "p_holm": float(p_holm[j]), "p_romano_wolf": float(p_rw[j])})
    return sorted(rows, key=lambda r: (r["cost"], r["arm"]))


def economic_size(res: dict, vs: str = "delta", notional: float = 1_000_000.0) -> list[dict]:
    """CVaR and mean-P&L differences in money (per HK$1m notional) and % of average premium."""
    raw = res["raw"]
    prem = float(np.mean(res["premium"]))
    rows = []
    for key, a in raw.items():
        arm, cost = key.split("|")
        if arm == vs or f"{vs}|{cost}" not in raw:
            continue
        b = raw[f"{vs}|{cost}"]
        common = a["mask"] & b["mask"]                   # compare the same windows only
        pa, pb = a["pnl"][:, common[a["mask"]]], b["pnl"][:, common[b["mask"]]]
        d_cvar = seed_avg_cvar(pa) - seed_avg_cvar(pb)
        d_mean = float(pa.mean() - pb.mean())
        scale = notional / 100.0                        # P&L is per HK$100 of underlying
        rows.append({"arm": arm, "cost": float(cost), "cvar_diff_per_100": d_cvar,
                     "cvar_diff_hkd_per_1m": d_cvar * scale, "cvar_diff_pct_premium": 100 * d_cvar / prem,
                     "mean_pnl_diff_hkd_per_1m": d_mean * scale, "avg_premium_per_100": prem})
    return sorted(rows, key=lambda r: (r["cost"], r["arm"]))
