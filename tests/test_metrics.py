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


def test_paired_bootstrap_detects_and_rejects():
    from deephedge.metrics import paired_bootstrap_cvar_diff
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 4000)
    same = paired_bootstrap_cvar_diff(base, base + rng.normal(0, 0.01, 4000), n_boot=400, seed=1)
    assert same["ci_low"] < 0 < same["ci_high"]          # no real difference -> CI covers 0
    worse = paired_bootstrap_cvar_diff(base, base - 1.0, n_boot=400, seed=1)  # b loses 1 more everywhere
    assert worse["diff"] < 0 and worse["ci_high"] < 0     # a has lower CVaR than b, significantly
    assert np.isclose(worse["diff"], -1.0, atol=0.05)
    assert 0 <= worse["p_a_not_better"] <= 1


def test_paired_bootstrap_clustered_wider_with_common_shocks():
    from deephedge.metrics import paired_bootstrap_cvar_diff
    rng = np.random.default_rng(0)
    n_groups, per = 60, 30
    groups = np.repeat(np.arange(n_groups), per)
    shock = rng.normal(0, 1, n_groups)[groups]          # windows in the same month share a shock
    a = shock + rng.normal(0, 0.3, len(groups))
    b = a + 0.2 * shock                                  # difference driven by the common shock
    plain = paired_bootstrap_cvar_diff(a, b, n_boot=400, seed=2)
    clus = paired_bootstrap_cvar_diff(a, b, n_boot=400, seed=2, clusters=groups)
    assert (clus["ci_high"] - clus["ci_low"]) > (plain["ci_high"] - plain["ci_low"])
    assert np.isclose(plain["diff"], clus["diff"])
