import numpy as np
import pandas as pd
import pytest

from deephedge.data.realdata import build_windows
from deephedge.sim.generators import (
    GarchT, fit_garch_t, gbm_returns, sample_mixture, simulate_garch_t,
    stationary_bootstrap, windows_from_returns,
)


def test_windows_from_returns_matches_build_windows():
    rng = np.random.default_rng(0)
    lr = rng.normal(0, 0.02, 84)
    prices = 50 * np.exp(np.r_[0.0, np.cumsum(lr)])          # 85 prices
    s = pd.Series(prices, index=pd.bdate_range("2015-01-01", periods=85))
    w = build_windows({"A": s})
    paths, tv = windows_from_returns(lr[None, :])
    assert len(w) == 1
    assert np.allclose(paths, w.paths)
    assert np.allclose(tv, w.trailing_vol)


def test_gbm_returns_vol():
    rng = np.random.default_rng(1)
    r = gbm_returns(np.full(4000, 0.3), 84, rng)
    assert r.shape == (4000, 84)
    assert np.isclose(r.std() * np.sqrt(252), 0.3, rtol=0.02)


def test_bootstrap_values_from_source_and_block_length():
    rng = np.random.default_rng(2)
    src = np.arange(1000, dtype=float)          # distinct values reveal the blocks
    out = stationary_bootstrap(src, 300, 84, mean_block=10, rng=rng)
    assert out.shape == (300, 84)
    assert np.isin(out, src).all()
    cont = (np.diff(out, axis=1) == 1) | (np.diff(out, axis=1) == -999)   # next index, with wrap
    runs = 1 / (1 - cont.mean())                # mean block length
    assert 8 < runs < 12


def test_garch_unconditional_variance():
    m = GarchT(mu=0.0, omega=2e-6, alpha=0.08, beta=0.9, nu=6.0)
    r = simulate_garch_t(m, 2000, 500, np.random.default_rng(3))
    assert np.isclose(r.var(), m.omega / (1 - m.alpha - m.beta), rtol=0.1)


def test_garch_fit_recovers_simulated_params():
    m = GarchT(mu=0.0, omega=4e-6, alpha=0.1, beta=0.85, nu=5.0)
    r = simulate_garch_t(m, 1, 6000, np.random.default_rng(4))[0]
    f = fit_garch_t(r)
    assert abs(f.alpha - 0.1) < 0.05 and abs(f.beta - 0.85) < 0.07
    assert 3.0 < f.nu < 9.0
    assert f.alpha + f.beta < 1


def test_garch_fit_rejects_short_series():
    with pytest.raises(ValueError, match="500"):
        fit_garch_t(np.zeros(100))


def test_mixture_weights():
    rng = np.random.default_rng(5)
    per = {"big": np.ones(900), "small": np.ones(100) * 2}
    out = sample_mixture(per, 5000, lambda x, n, g: np.tile(x[:84], (n, 1)), rng)
    assert out.shape == (5000, 84)
    assert np.isclose((out[:, 0] == 1).mean(), 0.9, atol=0.02)


def test_garch_nu_floor():
    m = GarchT(mu=0.0, omega=4e-6, alpha=0.1, beta=0.85, nu=2.6)
    r = simulate_garch_t(m, 1, 4000, np.random.default_rng(6))[0]
    assert fit_garch_t(r, nu_min=4.0).nu >= 4.0
