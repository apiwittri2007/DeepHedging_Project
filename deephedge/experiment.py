"""Train the deep hedger across cost levels and compare to delta benchmarks."""

from __future__ import annotations

import numpy as np

from .delta import banded_delta_policy, delta_policy
from .gbm import simulate_gbm_paths
from .hedge import simulate_hedged_pnl
from .metrics import summary
from .network import make_policy
from .train import train_deep_hedger


def _evaluate(policy, paths, cost, K, r, sigma, T) -> np.ndarray:
    return np.asarray(simulate_hedged_pnl(paths, policy, K, r, sigma, T, cost))


def run_experiment(
    s0: float = 100.0,
    K: float = 100.0,
    r: float = 0.0,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 20,
    n_train: int = 4000,
    n_eval: int = 8000,
    cost_grid=(0.0, 0.0005, 0.001, 0.002, 0.005),
    band: float = 0.05,
    layer_sizes=(4, 32, 32, 1),
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 0.01,
    seed: int = 0,
    focus_cost: float = 0.002,
) -> dict:
    """Return rows of metrics plus extras (focus-cost P&L, loss curve, example path)."""
    train_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_train, seed=seed)
    eval_paths = simulate_gbm_paths(s0, r, sigma, T, n_steps, n_eval, seed=seed + 1)
    common = dict(K=K, r=r, sigma=sigma, T=T)

    rows: list[dict] = []
    extras: dict = {}
    for cost in cost_grid:
        params, history = train_deep_hedger(
            train_paths, cost_rate=cost, layer_sizes=list(layer_sizes),
            epochs=epochs, batch_size=batch_size, lr=lr, seed=seed, **common,
        )
        policies = {
            "deep": make_policy(params["net"]),
            "delta": delta_policy(K, r, sigma),
            "banded_delta": banded_delta_policy(K, r, sigma, band),
            "no_hedge": lambda f: np.zeros(f.shape[0]),
        }
        pnls = {name: _evaluate(p, eval_paths, cost, **common) for name, p in policies.items()}
        for name, pnl in pnls.items():
            rows.append(dict(cost=cost, strategy=name, **summary(pnl)))

        if np.isclose(cost, focus_cost):
            extras["focus_cost"] = cost
            extras["pnl"] = {k: v for k, v in pnls.items()}
            extras["history"] = history
            extras["example"] = _example_holdings(eval_paths[0], params, K, r, sigma, T, n_steps)

    return {"rows": rows, "extras": extras}


def _example_holdings(path, params, K, r, sigma, T, n_steps):
    """Deep vs delta holding along one sample path (for a plot)."""
    import autograd.numpy as anp

    from .gbm import bs_call_delta
    from .network import mlp_forward

    dt = T / n_steps
    deep, delta, spot = [], [], []
    deep_shares = 0.0
    for k in range(n_steps):
        s_k = path[k]
        ttm = T - k * dt
        d = float(bs_call_delta(s_k, K, r, sigma, max(ttm, 1e-8)))
        feats = anp.array([[ttm, np.log(s_k / K), deep_shares, d]])
        deep_shares = float(mlp_forward(params["net"], feats)[0])
        deep.append(deep_shares)
        delta.append(d)
        spot.append(float(s_k))
    return {"spot": spot, "deep": deep, "delta": delta}
