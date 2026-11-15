"""Which training data makes a better deep hedger on real Hong Kong stocks?

Walk-forward protocol: for each test year Y, every arm may use only prices dated
before Y-01-01 (real training windows must also *end* before then). Each arm's
hedger is trained per (fold, cost, seed) and evaluated on that year's real,
non-overlapping test windows; P&L is pooled over all test years, CVaR is taken
per seed and averaged, and arms are compared with a month-clustered paired
bootstrap against BS delta and against A1 (GBM with estimated vol).

Arms (see the design doc): A0 GBM with known vol (the published method), A1 GBM
with estimated vol, A2 GARCH(1,1)-t, A3 stationary block bootstrap, A4 real
overlapping windows, A5 A1 fine-tuned on A4. Benchmarks: BS delta, banded delta,
no hedge, and L (least-squares linear correction to delta, fitted on A4's data).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from deephedge.data.realdata import TRADING_DAYS, Windows, build_windows, returns_before
from deephedge.hedging.delta import feature_banded_policy, feature_delta_policy
from deephedge.hedging.hedge import simulate_hedged_pnl
from deephedge.hedging.metrics import cvar, paired_bootstrap_cvar_diff
from deephedge.models.linear import fit_linear_correction, linear_policy
from deephedge.models.network import make_policy
from deephedge.models.train import train_deep_hedger
from deephedge.sim.gbm import simulate_gbm_paths
from deephedge.sim.generators import (
    MIN_GARCH_OBS, N_DAYS, VOL_LOOKBACK, fit_garch_t, gbm_returns, sample_mixture, simulate_garch_t,
    stationary_bootstrap, windows_from_returns,
)

ARMS = ("A0_gbm_true_vol", "A1_gbm", "A2_garch_t", "A3_bootstrap", "A4_real", "A5_finetune")
# Sensitivity arm added after the pre-registered run: GARCH-t with Student-t dof >= 4
# (finite shock kurtosis), motivated only by the pre-year realism check.
SENSITIVITY = ("A2b_garch_nu4",)
ALL_ARMS = ARMS + SENSITIVITY
BENCHMARKS = ("delta", "banded_delta", "no_hedge", "L_linear")
K, T, HORIZON = 100.0, 21 / TRADING_DAYS, 21
SYNTHETIC = ("A0_gbm_true_vol", "A1_gbm", "A2_garch_t", "A3_bootstrap", "A2b_garch_nu4")


def _no_hedge(feats):
    return feats[:, 0] * 0.0


# ---------------------------------------------------------------- fold inputs

def fold_inputs(closes: dict, year: int) -> dict:
    """Everything fold ``year`` may use, built from prices dated before ``year``-01-01."""
    cut = f"{year}-01-01"
    every = build_windows(closes)                                     # published test windows
    s = every.start
    test = every.subset((s >= np.datetime64(cut)) & (s < np.datetime64(f"{year + 1}-01-01")))
    return {
        "year": year,
        "test": test,
        "real_train": build_windows(closes, step=1, end_before=cut),
        "returns": returns_before(closes, cut),
    }


def _training_sets(f: dict, n_train: int, mean_block: float, seed: int) -> dict:
    """(paths, trader vol) per available arm for one fold and seed; None = unavailable."""
    rng = np.random.default_rng([f["year"], seed, 7])
    real: Windows = f["real_train"]
    rets = {k: v for k, v in f["returns"].items() if len(v) >= N_DAYS}
    vols = real.trailing_vol
    sets = {a: None for a in ALL_ARMS}
    if len(real) and len(vols):
        sig = rng.choice(vols, n_train)
        sets["A0_gbm_true_vol"] = (simulate_gbm_paths(100.0, 0.0, sig, T, HORIZON, n_train,
                                                       seed=int(rng.integers(2**31))), sig)
        sets["A1_gbm"] = windows_from_returns(gbm_returns(rng.choice(vols, n_train), N_DAYS, rng))
        sets["A4_real"] = (real.paths, real.trailing_vol)
        sets["A5_finetune"] = sets["A4_real"]
    if rets:
        sets["A3_bootstrap"] = windows_from_returns(sample_mixture(
            rets, n_train, lambda r, k, g: stationary_bootstrap(r, k, N_DAYS, mean_block, g), rng))
    garch = {k: v for k, v in f["returns"].items() if len(v) >= MIN_GARCH_OBS}
    if garch:
        if "_garch" not in f:                    # fit once per fold, reuse across seeds
            f["_garch"] = {k: fit_garch_t(v) for k, v in garch.items()}
        sets["A2_garch_t"] = windows_from_returns(_garch_draw(f, n_train, rng))
        if "_garch4" not in f:
            f["_garch4"] = {k: fit_garch_t(v, nu_min=4.0) for k, v in garch.items()}
        sets["A2b_garch_nu4"] = windows_from_returns(_garch_draw(f, n_train, rng, key="_garch4"))
    return {a: _clean(v) for a, v in sets.items()}


def _clean(s):
    if s is None:
        return None
    p, v = s
    ok = np.isfinite(p).all(axis=1) & np.isfinite(v) & (v > 0.01)
    return (p[ok], v[ok]) if ok.any() else None


# ---------------------------------------------------------------- jobs

def _job(args):
    """Train one arm (A1 jobs also fine-tune into A5) and return test P&L per arm."""
    arm, cost, seed, train, finetune, test_paths, test_vol, cfg = args
    common = dict(K=K, r=0.0, T=T, cost_rate=cost, layer_sizes=list(cfg["layer_sizes"]),
                  batch_size=cfg["batch_size"], seed=seed)
    params, _ = train_deep_hedger(train[0], sigma=train[1], epochs=cfg["epochs"], lr=cfg["lr"], **common)
    ev = lambda th: np.asarray(simulate_hedged_pnl(test_paths, make_policy(th["net"]), K, 0.0,
                                                   test_vol, T, cost))
    out = {arm: ev(params)}
    if finetune is not None:
        ft, _ = train_deep_hedger(finetune[0], sigma=finetune[1], epochs=cfg["finetune_epochs"],
                                  lr=cfg["finetune_lr"], init=params, **common)
        out["A5_finetune"] = ev(ft)
    return out


def _realism(x: np.ndarray) -> dict:
    """Stylised facts of an (n, 84) return array, as the hedger sees them.

    Tails are measured on the 21 hedge-day returns divided by the trader's own
    63-day trailing vol (so mixing calm and wild stocks does not count as fat
    tails); clustering is the within-path autocorrelation of |returns|.
    """
    trail = x[:, :VOL_LOOKBACK].std(axis=1, ddof=1, keepdims=True)
    z = (x[:, VOL_LOOKBACK:] / trail).ravel()
    z = z[np.isfinite(z)]
    zc = z - z.mean()
    kurt = float((zc**4).mean() / (zc**2).mean() ** 2 - 3.0)
    a = np.abs(x) - np.abs(x).mean(axis=1, keepdims=True)
    acf = [float((a[:, :-k] * a[:, k:]).mean() / (a**2).mean()) for k in range(1, 11)]
    tail = np.sort(np.abs(zc))[::-1]
    k = max(10, int(0.025 * len(tail)))
    hill = float(1.0 / np.mean(np.log(tail[:k] / tail[k])))
    return {"excess_kurtosis": kurt, "acf_abs_1_10": float(np.mean(acf)), "hill_index": hill}


def fold_realism(f: dict, sets: dict, n_train: int, mean_block: float, seed: int) -> list[dict]:
    """Realism rows for one fold: real pre-year data and each available generator."""
    rows = []
    real_lr = _real_blocks(f["returns"])
    if len(real_lr):
        rows.append({"year": f["year"], "source": "real", **_realism(real_lr)})
    for a in SYNTHETIC[1:]:                      # A0 has the same returns as A1 (GBM); unrun arms are None
        if sets[a] is not None:
            rows.append({"year": f["year"], "source": a,
                         **_realism(_synthetic_lr(f, a, n_train, mean_block, seed))})
    return rows


def _real_blocks(returns: dict) -> np.ndarray:
    blocks = [v[i:i + N_DAYS] for v in returns.values() for i in range(0, len(v) - N_DAYS + 1, HORIZON)]
    return np.array(blocks) if blocks else np.zeros((0, N_DAYS))


# ---------------------------------------------------------------- runner

def run_training_data_study(
    closes: dict,
    years=tuple(range(2016, 2027)),
    cost_grid=(0.0, 0.0005, 0.001, 0.0015, 0.0025, 0.005),
    focus_cost: float = 0.0015,
    seeds=(0, 1, 2),
    n_train: int = 6000,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 256,
    finetune_epochs: int = 50,
    finetune_lr: float = 0.002,
    mean_block: float = 10.0,
    layer_sizes=(4, 32, 32, 1),
    band: float = 0.05,
    n_boot: int = 2000,
    n_jobs: int | None = None,
    log=None,
    arms=ARMS,
) -> dict:
    cfg = dict(epochs=epochs, lr=lr, batch_size=batch_size, finetune_epochs=finetune_epochs,
               finetune_lr=finetune_lr, layer_sizes=tuple(layer_sizes))
    log = log or (lambda *_: None)
    folds = [fold_inputs(closes, y) for y in years]
    folds = [f for f in folds if len(f["test"])]

    jobs, keys, fold_sizes, realism = [], [], [], []
    bench = {}                                   # (arm, cost) -> list of per-fold pnl
    for fi, f in enumerate(folds):
        test = f["test"]
        sets_by_seed = {s: _training_sets(f, n_train, mean_block, s) for s in seeds}
        first = sets_by_seed[seeds[0]]
        unavailable = sorted(a for a in arms if first[a] is None)
        fold_sizes.append({"year": f["year"], "n_test": len(test), "n_real_train": len(f["real_train"]),
                           "n_stocks_returns": len(f["returns"]),
                           "n_stocks_garch": len(f.get("_garch", {})), "unavailable": unavailable})
        realism += fold_realism(f, first, n_train, mean_block, seeds[0])
        lin = None
        if first["A4_real"] is not None and len(first["A4_real"][0]) >= 10:
            lin = fit_linear_correction(*first["A4_real"], K=K, T=T)
        for cost in cost_grid:
            pols = {"delta": feature_delta_policy(), "banded_delta": feature_banded_policy(band),
                    "no_hedge": _no_hedge}
            if lin is not None:
                pols["L_linear"] = linear_policy(lin)
            for name, pol in pols.items():
                pnl = np.asarray(simulate_hedged_pnl(test.paths, pol, K, 0.0, test.trailing_vol, T, cost))
                bench.setdefault((name, cost), {})[fi] = pnl
            for s in seeds:
                sets = sets_by_seed[s]
                for a in arms:
                    if a == "A5_finetune" or sets[a] is None:
                        continue
                    ft = sets["A5_finetune"] if a == "A1_gbm" else None
                    job_seed = 10_000 * s + 100 * fi + ALL_ARMS.index(a)
                    jobs.append((a, cost, job_seed, sets[a], ft, test.paths, test.trailing_vol, cfg))
                    keys.append((fi, cost, s))
    log(f"{len(jobs)} training jobs over {len(folds)} folds")

    deep = {}                                    # (arm, cost, seed) -> {fold: pnl}
    n_jobs = n_jobs or os.cpu_count() or 1
    results = (map(_job, jobs) if n_jobs == 1 else
               ProcessPoolExecutor(n_jobs).map(_job, jobs, chunksize=1))
    for i, ((fi, cost, s), out) in enumerate(zip(keys, results)):
        for arm, pnl in out.items():
            deep.setdefault((arm, cost, s), {})[fi] = pnl
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(jobs)} jobs done")

    return _assemble(folds, cost_grid, focus_cost, seeds, deep, bench, fold_sizes, realism, n_boot)


def _synthetic_lr(f, arm, n_train, mean_block, seed):
    """Full 84-day returns of a generator, for realism statistics (same rng as training)."""
    rng = np.random.default_rng([f["year"], seed, 11])
    vols = f["real_train"].trailing_vol
    if arm in ("A0_gbm_true_vol", "A1_gbm"):
        return gbm_returns(rng.choice(vols, n_train), N_DAYS, rng)
    rets = {k: v for k, v in f["returns"].items() if len(v) >= N_DAYS}
    if arm == "A3_bootstrap":
        return sample_mixture(rets, n_train, lambda r, k, g: stationary_bootstrap(r, k, N_DAYS, mean_block, g), rng)
    return _garch_draw(f, n_train, rng, key="_garch4" if arm == "A2b_garch_nu4" else "_garch")


def _garch_draw(f, n, rng, key="_garch"):
    """GARCH-t paths per stock, started at that stock's sample variance.

    Starting at the sample variance (not the model's unconditional variance)
    keeps near-integrated fits (alpha + beta close to 1, where the unconditional
    variance explodes) on the scale the stock actually traded at.
    """
    items = {k: (m, float(np.var(f["returns"][k]))) for k, m in f[key].items()}
    weights = {k: len(f["returns"][k]) for k in items}
    return sample_mixture(items, n, lambda mv, k, g: simulate_garch_t(mv[0], k, N_DAYS, g, h0=mv[1]),
                          rng, weights=weights)


def _assemble(folds, cost_grid, focus_cost, seeds, deep, bench, fold_sizes, realism, n_boot):
    all_folds = list(range(len(folds)))
    months = np.concatenate([f["test"].start.astype("datetime64[M]") for f in folds])
    years_arr = np.concatenate([np.full(len(f["test"]), f["year"]) for f in folds])
    offsets = np.cumsum([0] + [len(f["test"]) for f in folds])

    def pooled(by_fold):
        """Concatenate fold P&L; returns (pnl, mask of covered test windows)."""
        mask = np.zeros(offsets[-1], dtype=bool)
        parts = []
        for fi in all_folds:
            if fi in by_fold:
                parts.append(by_fold[fi])
                mask[offsets[fi]:offsets[fi + 1]] = True
        return np.concatenate(parts), mask

    series = {}                                  # (arm, cost) -> (pnl 2-D seeds x n, mask)
    for (name, cost), by_fold in bench.items():
        p, m = pooled(by_fold)
        series[(name, cost)] = (p[None, :], m)
    for arm in ALL_ARMS:
        for cost in cost_grid:
            per_seed = [deep.get((arm, cost, s)) for s in seeds]
            if any(x is None for x in per_seed):
                continue
            stacked = [pooled(x) for x in per_seed]
            if any((m != stacked[0][1]).any() for _, m in stacked):
                continue
            series[(arm, cost)] = (np.stack([p for p, _ in stacked]), stacked[0][1])

    summary, significance, by_year = [], [], []
    for (arm, cost), (p, m) in sorted(series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cv = np.array([cvar(row) for row in p])
        summary.append({"arm": arm, "cost": cost, "cvar_mean": float(cv.mean()),
                        "cvar_sd": float(cv.std(ddof=1)) if len(cv) > 1 else 0.0,
                        "mean": float(p.mean()), "n": int(m.sum()), "n_seeds": len(cv)})
        if arm in ALL_ARMS or arm == "L_linear":
            for vs in ("delta", "A1_gbm"):
                if vs == arm or (vs, cost) not in series:
                    continue
                q, mq = series[(vs, cost)]
                common = m & mq
                a_ = p[:, common[m]]
                b_ = q[:, common[mq]]
                res = paired_bootstrap_cvar_diff(a_, b_, n_boot=n_boot, seed=0, clusters=months[common])
                significance.append({"arm": arm, "cost": cost, "vs": vs, "diff": res["diff"],
                                     "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                                     "p": res["p_a_not_better"]})
        if cost == focus_cost:
            yrs = years_arr[m]
            for y in np.unique(yrs):
                sel = yrs == y
                by_year.append({"arm": arm, "year": int(y),
                                "cvar_mean": float(np.mean([cvar(row[sel]) for row in p]))})

    return {"summary": summary, "significance": significance, "by_year": by_year,
            "realism": realism, "fold_sizes": fold_sizes, "n_test": int(offsets[-1]),
            "focus_cost": focus_cost}
