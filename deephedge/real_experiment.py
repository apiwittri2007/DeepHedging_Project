"""Out-of-sample test of the deep hedger on real Hong Kong stock paths.

Protocol (no look-ahead anywhere):
1. Split windows by start date: before ``split`` = calibration, on/after = test.
2. Train the deep hedger on *simulated* GBM paths whose per-path volatility is
   drawn from the calibration windows' trailing-vol distribution. No real path
   is ever trained on.
3. Evaluate deep, BS-delta, banded-delta and no-hedge on the real test windows,
   each priced and delta-hedged with its own trailing vol — and, for contrast, on
   fresh simulated paths with the same vol distribution ("sim").
"""

from __future__ import annotations

import numpy as np

from .delta import feature_banded_policy, feature_delta_policy
from .gbm import simulate_gbm_paths
from .hedge import simulate_hedged_pnl
from .metrics import cvar, summary
from .network import make_policy
from .realdata import TRADING_DAYS, Windows
from .train import train_deep_hedger

PERIODS = {
    "COVID crash": ("2020-02-01", "2020-06-30"),
    "Tech crackdown": ("2021-02-01", "2022-10-31"),
    "2023 onwards": ("2023-01-01", "2100-01-01"),
}


def _excess_kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x**4).mean() / (x**2).mean() ** 2 - 3.0)


def _stylized(w: Windows) -> dict:
    lr = np.diff(np.log(w.paths), axis=1)                    # (n, horizon) daily log returns
    daily_vol = (w.trailing_vol / np.sqrt(TRADING_DAYS))[:, None]
    std_r = (lr / daily_vol).ravel()                         # returns in units of the trader's vol
    return {
        "excess_kurtosis_raw": _excess_kurtosis(lr.ravel()),
        "excess_kurtosis_std": _excess_kurtosis(std_r),
        "std_returns": std_r,
        "share_beyond_4sd": float((np.abs(std_r) > 4).mean()),
        "normal_share_beyond_4sd": 6.33e-5,
    }


def run_real_experiment(
    windows: Windows,
    split: str = "2016-01-01",
    cost_grid=(0.0, 0.0005, 0.001, 0.0015, 0.0025, 0.005),
    focus_cost: float = 0.0015,
    n_train: int = 6000,
    n_sim_eval: int = 8000,
    epochs: int = 200,
    layer_sizes=(4, 32, 32, 1),
    batch_size: int = 256,
    lr: float = 0.01,
    band: float = 0.05,
    seed: int = 0,
) -> dict:
    split_d = np.datetime64(split)
    calib = windows.subset(windows.start < split_d)
    test = windows.subset(windows.start >= split_d)
    if len(calib) == 0:
        raise ValueError(f"no calibration windows before {split}; cannot calibrate the vol distribution")
    if len(test) == 0:
        raise ValueError(f"no test windows on/after {split}")

    horizon = windows.paths.shape[1] - 1
    T = horizon / TRADING_DAYS
    K = 100.0
    rng = np.random.default_rng(seed)
    train_sig = rng.choice(calib.trailing_vol, size=n_train, replace=True)
    train_paths = simulate_gbm_paths(100.0, 0.0, train_sig, T, horizon, n_train, seed=seed)
    sim_sig = rng.choice(calib.trailing_vol, size=n_sim_eval, replace=True)
    sim_paths = simulate_gbm_paths(100.0, 0.0, sim_sig, T, horizon, n_sim_eval, seed=seed + 1)
    datasets = {"real": (test.paths, test.trailing_vol), "sim": (sim_paths, sim_sig)}

    rows, periods = [], []
    focus_pnl, history, zero_cost_delta = {}, [], None
    for cost in cost_grid:
        params, hist = train_deep_hedger(
            train_paths, K=K, r=0.0, sigma=train_sig, T=T, cost_rate=cost,
            layer_sizes=list(layer_sizes), epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
        )
        policies = {
            "deep": make_policy(params["net"]),
            "delta": feature_delta_policy(),
            "banded_delta": feature_banded_policy(band),
            "no_hedge": lambda f: np.zeros(f.shape[0]),
        }
        for ds, (paths, sig) in datasets.items():
            for name, pol in policies.items():
                pnl = np.asarray(simulate_hedged_pnl(paths, pol, K, 0.0, sig, T, cost))
                rows.append(dict(dataset=ds, cost=cost, strategy=name, **summary(pnl)))
                if ds == "real" and np.isclose(cost, focus_cost):
                    focus_pnl[name] = pnl
                if ds == "real" and cost == 0.0 and name == "delta":
                    zero_cost_delta = pnl
        if np.isclose(cost, focus_cost):
            history = hist

    for pname, (a, b) in {"All test": (split, "2100-01-01"), **PERIODS}.items():
        mask = (test.start >= np.datetime64(a)) & (test.start <= np.datetime64(b))
        for name, pnl in focus_pnl.items():
            sel = pnl[mask]
            periods.append(dict(period=pname, strategy=name, n=int(mask.sum()),
                                cvar95=cvar(sel, 0.95) if len(sel) else float("nan"),
                                mean=float(sel.mean()) if len(sel) else float("nan")))

    if zero_cost_delta is None:  # cost grid without 0: evaluate delta at zero cost for the vol-gap analysis
        zero_cost_delta = np.asarray(simulate_hedged_pnl(
            test.paths, feature_delta_policy(), K, 0.0, test.trailing_vol, T, 0.0))

    return {
        "rows": rows,
        "periods": periods,
        "focus_cost": focus_cost,
        "focus_pnl": focus_pnl,
        "history": history,
        "misspec": {"gap": test.trailing_vol - test.realised_vol, "pnl": zero_cost_delta},
        "stylized": _stylized(test),
        "sample": {"real": test.paths[:: max(1, len(test) // 60)][:60], "sim": sim_paths[:60]},
        "calib_vols": calib.trailing_vol,
        "n": {"calib": len(calib), "test": len(test), "dropped": windows.n_dropped,
              "stocks": int(len(set(windows.ticker)))},
    }
