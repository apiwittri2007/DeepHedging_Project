import numpy as np
import pandas as pd
import pytest

from deephedge.real_experiment import run_real_experiment
from deephedge.realdata import build_windows

STRATS = {"deep", "delta", "banded_delta", "no_hedge"}


@pytest.fixture(scope="module")
def windows():
    closes = {}
    for i in range(4):
        rng = np.random.default_rng(i)
        n = 2600  # ~2010-01 .. 2019-12
        p = 50 * np.exp(np.cumsum(rng.normal(0, 0.025, n)))
        closes[f"S{i}"] = pd.Series(p, index=pd.bdate_range("2010-01-04", periods=n))
    return build_windows(closes)


@pytest.fixture(scope="module")
def result(windows):
    return run_real_experiment(windows, split="2016-01-01", cost_grid=(0.0, 0.0015),
                               focus_cost=0.0015, n_train=800, n_sim_eval=800,
                               epochs=25, layer_sizes=(4, 8, 1), seed=0)


def test_rows_structure(result):
    rows = result["rows"]
    assert {r["dataset"] for r in rows} == {"real", "sim"}
    assert {r["strategy"] for r in rows} == STRATS
    assert {r["cost"] for r in rows} == {0.0, 0.0015}
    assert len(rows) == 2 * 2 * 4
    assert result["n"]["calib"] > 0 and result["n"]["test"] > 0


def test_periods_present(result):
    names = {p["period"] for p in result["periods"]}
    assert "All test" in names and "Tech crackdown" in names
    assert {p["strategy"] for p in result["periods"]} == STRATS


def test_extras(result):
    assert set(result["focus_pnl"]) == STRATS
    assert len(result["misspec"]["gap"]) == len(result["misspec"]["pnl"]) == result["n"]["test"]
    assert np.isfinite(result["stylized"]["excess_kurtosis_raw"])


def test_deep_beats_no_hedge(result):
    real = {r["strategy"]: r["cvar95"] for r in result["rows"]
            if r["dataset"] == "real" and r["cost"] == 0.0015}
    assert real["deep"] < real["no_hedge"]


def test_empty_calibration_raises(windows):
    with pytest.raises(ValueError, match="calibration"):
        run_real_experiment(windows, split="2000-01-01", cost_grid=(0.0,), focus_cost=0.0,
                            n_train=100, epochs=1, layer_sizes=(4, 4, 1))
