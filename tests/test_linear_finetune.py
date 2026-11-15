import numpy as np

from deephedge.hedging.delta import feature_delta_policy
from deephedge.hedging.hedge import simulate_hedged_pnl
from deephedge.hedging.metrics import cvar, paired_bootstrap_cvar_diff
from deephedge.models.linear import fit_linear_correction, linear_policy
from deephedge.models.train import train_deep_hedger
from deephedge.sim.gbm import simulate_gbm_paths

T = 21 / 252


def _paths(n=3000, seed=0):
    return simulate_gbm_paths(100, 0.0, 0.3, T, 21, n, seed)


def test_linear_zero_coef_equals_delta():
    p = _paths(500)
    a = simulate_hedged_pnl(p, linear_policy(np.zeros(3)), 100, 0.0, 0.3, T, 0.001)
    b = simulate_hedged_pnl(p, feature_delta_policy(), 100, 0.0, 0.3, T, 0.001)
    assert np.allclose(a, b)


def test_linear_fit_reduces_variance_in_sample():
    rng = np.random.default_rng(1)
    p = _paths(3000)
    sig = rng.uniform(0.2, 0.45, len(p))          # misspecified vol -> delta is not optimal
    coef = fit_linear_correction(p, sig)
    assert coef.shape == (3,)
    a = simulate_hedged_pnl(p, linear_policy(coef), 100, 0.0, sig, T, 0.0)
    b = simulate_hedged_pnl(p, feature_delta_policy(), 100, 0.0, sig, T, 0.0)
    assert np.var(a) <= np.var(b) + 1e-12


def test_finetune_zero_epochs_returns_init():
    p = _paths(300)
    init, _ = train_deep_hedger(p, 100, 0.0, 0.3, T, 0.001, layer_sizes=[4, 8, 1], epochs=3, seed=0)
    out, hist = train_deep_hedger(p, 100, 0.0, 0.3, T, 0.001, layer_sizes=[4, 8, 1], epochs=0, init=init)
    assert hist == []
    for (W0, b0), (W1, b1) in zip(init["net"], out["net"]):
        assert np.allclose(W0, W1) and np.allclose(b0, b1)
    assert np.isclose(init["w"], out["w"])


def test_bootstrap_2d_diff_is_mean_of_seed_cvars():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, (3, 400))
    b = rng.normal(0, 1.2, 400)
    res = paired_bootstrap_cvar_diff(a, b, n_boot=200)
    expected = np.mean([cvar(x) for x in a]) - cvar(b)
    assert np.isclose(res["diff"], expected)
    assert res["ci_low"] < res["diff"] < res["ci_high"]
