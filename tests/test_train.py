import numpy as np

from deephedge.gbm import simulate_gbm_paths
from deephedge.hedge import simulate_hedged_pnl
from deephedge.metrics import cvar
from deephedge.network import init_params, make_policy
from deephedge.train import train_deep_hedger

ARGS = dict(K=100, r=0.0, sigma=0.2, T=1.0)


def eval_cvar(policy, paths, cost):
    pnl = np.asarray(simulate_hedged_pnl(paths, policy, cost_rate=cost, **ARGS))
    return cvar(pnl, 0.95)


def test_training_reduces_tail_risk():
    train_paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 3000, seed=0)
    eval_paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 3000, seed=99)
    layers = [4, 16, 16, 1]
    cost = 0.001

    init = make_policy(init_params(layers, seed=0))
    cvar_before = eval_cvar(init, eval_paths, cost)

    params, history = train_deep_hedger(
        train_paths, cost_rate=cost, alpha=0.95, layer_sizes=layers,
        epochs=120, batch_size=256, lr=0.01, seed=0, **ARGS,
    )
    cvar_after = eval_cvar(make_policy(params["net"]), eval_paths, cost)

    assert len(history) == 120
    assert cvar_after < cvar_before            # learned to cut tail risk
    assert history[-1] < history[0]            # training loss fell


def test_returns_learnable_w():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 1000, seed=1)
    params, _ = train_deep_hedger(paths, cost_rate=0.0, layer_sizes=[4, 8, 1],
                                  epochs=10, batch_size=128, lr=0.01, seed=1, **ARGS)
    assert "net" in params and "w" in params
