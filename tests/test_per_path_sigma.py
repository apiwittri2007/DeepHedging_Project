import autograd.numpy as anp
import numpy as np
import pytest

from deephedge.hedging.delta import feature_banded_policy, feature_delta_policy
from deephedge.sim.gbm import bs_call_delta, simulate_gbm_paths
from deephedge.hedging.hedge import simulate_hedged_pnl
from deephedge.models.train import train_deep_hedger


def test_gbm_per_path_sigma_dispersion():
    n = 20000
    sig = np.r_[np.full(n, 0.1), np.full(n, 0.6)]
    p = simulate_gbm_paths(100, 0.0, sig, 1.0, 20, 2 * n, seed=0)
    lr = np.log(p[:, -1] / p[:, 0])
    assert np.isclose(lr[:n].std(), 0.1, rtol=0.03)
    assert np.isclose(lr[n:].std(), 0.6, rtol=0.03)


def test_engine_sigma_array_equals_scalar_when_constant():
    p = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 300, seed=1)
    pol = feature_delta_policy()
    a = np.asarray(simulate_hedged_pnl(p, pol, 100, 0.0, 0.2, 1.0, 0.001))
    b = np.asarray(simulate_hedged_pnl(p, pol, 100, 0.0, np.full(300, 0.2), 1.0, 0.001))
    assert np.allclose(a, b)


def test_wrong_length_sigma_raises():
    p = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 5, 10, seed=2)
    with pytest.raises(ValueError, match="sigma"):
        simulate_hedged_pnl(p, feature_delta_policy(), 100, 0.0, np.full(7, 0.2), 1.0, 0.0)
    with pytest.raises(ValueError, match="sigma"):
        simulate_gbm_paths(100, 0.0, np.full(7, 0.2), 1.0, 5, 10, seed=2)


def test_feature_delta_policy_matches_bs_delta():
    feats = anp.array([[0.5, 0.1, 0.0, float(bs_call_delta(100 * np.exp(0.1), 100, 0.0, 0.3, 0.5))]])
    assert np.isclose(feature_delta_policy()(feats)[0], bs_call_delta(100 * np.exp(0.1), 100, 0.0, 0.3, 0.5))
    banded = feature_banded_policy(0.1)
    near = anp.array([[0.5, 0.0, 0.50, 0.55]])
    far = anp.array([[0.5, 0.0, 0.10, 0.55]])
    assert np.isclose(banded(near)[0], 0.50) and np.isclose(banded(far)[0], 0.55)


def test_train_accepts_sigma_array():
    n = 600
    sig = np.random.default_rng(0).uniform(0.15, 0.5, n)
    p = simulate_gbm_paths(100, 0.0, sig, 21 / 252, 21, n, seed=3)
    params, hist = train_deep_hedger(p, K=100, r=0.0, sigma=sig, T=21 / 252, cost_rate=0.001,
                                     layer_sizes=[4, 8, 1], epochs=5, batch_size=128, seed=0)
    assert len(hist) == 5 and np.isfinite(hist).all()
