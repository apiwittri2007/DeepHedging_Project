import numpy as np
import pandas as pd

from deephedge.data.realdata import build_windows
from deephedge.experiments.training_data_study import ARMS, fold_inputs, run_training_data_study

QUICK = dict(cost_grid=(0.0, 0.0015), focus_cost=0.0015, seeds=(0,), n_train=300, epochs=3,
             finetune_epochs=2, layer_sizes=(4, 8, 1), n_jobs=1, n_boot=50)


def closes(start="2012-01-02", end="2017-12-29", n_stocks=2, seed=0):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_stocks):
        lr = rng.standard_t(5, len(idx)) * 0.015
        out[f"S{i}"] = pd.Series(30 * np.exp(np.cumsum(lr)), index=idx)
    return out


def test_fold_inputs_no_leakage():
    c = closes()
    f = fold_inputs(c, 2016)
    cut = np.datetime64("2016-01-01")
    assert (f["real_train"].end < cut).all()
    assert (f["test"].start >= cut).all() and (f["test"].start < np.datetime64("2017-01-01")).all()
    # returns_before uses only prices before the cut: changing 2016+ prices changes nothing
    c2 = {k: v.copy() for k, v in c.items()}
    for v in c2.values():
        v[v.index >= "2016-01-01"] *= 1.7
    g = fold_inputs(c2, 2016)
    for k in f["returns"]:
        assert np.array_equal(f["returns"][k], g["returns"][k])
    assert np.array_equal(f["real_train"].paths, g["real_train"].paths)


def test_runner_structure():
    res = run_training_data_study(closes(), years=(2016, 2017), **QUICK)
    arms = {r["arm"] for r in res["summary"]}
    assert set(ARMS) <= arms and {"delta", "banded_delta", "no_hedge", "L_linear"} <= arms
    assert len(res["summary"]) == len(arms) * 2
    sig = res["significance"]
    assert {r["vs"] for r in sig} == {"delta", "A1_gbm"}
    assert all(r["ci_low"] <= r["diff"] <= r["ci_high"] for r in sig)
    assert {r["year"] for r in res["by_year"]} == {2016, 2017}
    assert {r["source"] for r in res["realism"]} >= {"real", "A1_gbm", "A2_garch_t", "A3_bootstrap"}
    assert [r["year"] for r in res["fold_sizes"]] == [2016, 2017]


def test_unavailable_arm_reported():
    c = closes(start="2015-03-02", end="2016-12-30")     # < 500 returns before 2016
    res = run_training_data_study(c, years=(2016,), **QUICK)
    assert "A2_garch_t" in res["fold_sizes"][0]["unavailable"]
    assert not any(r["arm"] == "A2_garch_t" for r in res["summary"])
    assert any(r["arm"] == "A3_bootstrap" for r in res["summary"])


def test_pooled_test_set_equals_non_overlapping_windows():
    c = closes()
    res = run_training_data_study(c, years=(2016, 2017), **QUICK)
    w = build_windows(c)
    expected = int((w.start >= np.datetime64("2016-01-01")).sum())
    assert res["n_test"] == expected
    assert all(r["n"] == expected for r in res["summary"])
