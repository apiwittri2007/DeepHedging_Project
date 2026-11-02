import numpy as np
import pytest

from deephedge.gbm import bs_call_delta, bs_call_price, simulate_gbm_paths


def test_shape_and_start():
    p = simulate_gbm_paths(s0=100, mu=0.0, sigma=0.2, T=1.0, n_steps=50, n_paths=1000, seed=0)
    assert p.shape == (1000, 51)
    assert np.allclose(p[:, 0], 100.0)
    assert (p > 0).all()


def test_seed_reproducible():
    a = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 20, 100, seed=7)
    b = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 20, 100, seed=7)
    assert np.array_equal(a, b)


def test_terminal_mean_matches_drift():
    p = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 100, 200_000, seed=1)
    assert p[:, -1].mean() == pytest.approx(100 * np.exp(0.05), rel=0.01)


def test_bs_price_known_atm():
    # S=K=100, r=0, sigma=0.2, T=1: price = S*(2*Phi(sigma/2)-1) ~ 7.9656
    assert bs_call_price(100, 100, 0.0, 0.2, 1.0) == pytest.approx(7.9656, abs=1e-3)


def test_bs_price_at_expiry_is_payoff():
    assert bs_call_price(120, 100, 0.0, 0.2, 0.0) == pytest.approx(20.0)
    assert bs_call_price(90, 100, 0.0, 0.2, 0.0) == pytest.approx(0.0)


def test_bs_delta_bounds_and_limits():
    assert 0.0 <= bs_call_delta(100, 100, 0.0, 0.2, 1.0) <= 1.0
    assert bs_call_delta(100, 100, 0.0, 0.2, 1.0) == pytest.approx(0.5398, abs=1e-3)
    assert bs_call_delta(1000, 100, 0.0, 0.2, 0.5) == pytest.approx(1.0, abs=1e-6)
    assert bs_call_delta(10, 100, 0.0, 0.2, 0.5) == pytest.approx(0.0, abs=1e-6)


def test_bs_delta_at_expiry_is_indicator():
    assert bs_call_delta(120, 100, 0.0, 0.2, 0.0) == pytest.approx(1.0)
    assert bs_call_delta(90, 100, 0.0, 0.2, 0.0) == pytest.approx(0.0)
