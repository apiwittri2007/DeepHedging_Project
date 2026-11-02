import numpy as np

from deephedge.metrics import cvar, ru_cvar_objective, summary


def test_cvar_known_array():
    pnl = np.array([-5.0, -4, -3, -2, -1, 0, 1, 2, 3, 4])
    assert np.isclose(cvar(pnl, 0.9), 5.0)    # worst 10% = {-5}
    assert np.isclose(cvar(pnl, 0.8), 4.5)    # worst 20% = {-5,-4}


def test_ru_objective_minimised_equals_cvar():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0, 1, 5000)
    grid = np.linspace(-4, 4, 2001)
    best = min(ru_cvar_objective(pnl, w, 0.95) for w in grid)
    assert np.isclose(best, cvar(pnl, 0.95), atol=0.05)


def test_summary_keys():
    s = summary(np.array([-1.0, 0.0, 1.0, 2.0]))
    assert set(s) == {"mean", "std", "cvar95", "var95", "min"}
