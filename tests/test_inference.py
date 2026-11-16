import numpy as np

from deephedge.experiments.inference import (
    economic_size, holm, month_counts, multiple_testing, romano_wolf, seed_avg_cvar, weighted_cvar,
)
from deephedge.hedging.metrics import cvar


def test_weighted_cvar_matches_resampled_cvar():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    counts = rng.integers(0, 3, (5, 500))
    for b in range(5):
        assert np.isclose(weighted_cvar(x, counts[b:b + 1])[0], cvar(np.repeat(x, counts[b])))
    assert np.isclose(seed_avg_cvar(x[None]), cvar(x))


def test_month_counts_keep_months_together():
    months = np.repeat(np.array(["2020-01", "2020-02", "2020-03"]), [4, 2, 3])
    c = month_counts(months, 50, seed=1)
    assert c.shape == (50, 9)
    for m in np.unique(months):
        block = c[:, months == m]
        assert (block == block[:, :1]).all()          # same weight for every window in a month
    assert (np.array([len(np.unique(months))] * 50) == np.array(
        [c[b, [0, 4, 6]].sum() for b in range(50)])).all()   # number of months drawn is fixed


def test_holm_known_values():
    assert np.allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def test_romano_wolf_rejects_real_effect_not_null():
    rng = np.random.default_rng(2)
    t_star = rng.standard_normal((4000, 3))
    p = romano_wolf(np.array([-5.0, 0.0, 1.0]), t_star)
    assert p[0] < 0.01 and p[1] > 0.3 and p[2] > 0.5
    assert p[0] <= p[1] <= p[2]


def _res(effect):
    rng = np.random.default_rng(3)
    n = 1200
    months = np.repeat([f"2020-{i:02d}" for i in range(1, 13)], n // 12)
    base = rng.standard_t(4, n)
    raw = {"delta|0.0015": {"pnl": base[None].astype(np.float32), "mask": np.ones(n, bool)},
           "A1|0.0015": {"pnl": np.stack([base + effect * (base < -1.5) + rng.normal(0, .01, n)
                                          for _ in range(3)]).astype(np.float32), "mask": np.ones(n, bool)}}
    return {"raw": raw, "months": months, "premium": np.full(n, 4.0)}


def test_multiple_testing_detects_tail_improvement():
    row = multiple_testing(_res(1.0), n_boot=400)[0]
    assert row["diff"] < 0 and row["p_romano_wolf"] < 0.05 and row["ci_high"] < 0
    null = multiple_testing(_res(0.0), n_boot=400)[0]
    assert null["p_romano_wolf"] > 0.05


def test_economic_size_units():
    r = economic_size(_res(1.0))[0]
    assert np.isclose(r["cvar_diff_hkd_per_1m"], r["cvar_diff_per_100"] * 10_000)
    assert np.isclose(r["cvar_diff_pct_premium"], 100 * r["cvar_diff_per_100"] / 4.0)
