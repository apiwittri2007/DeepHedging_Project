from deephedge.__main__ import main
from deephedge.experiment import run_experiment


def test_quick_experiment_structure():
    res = run_experiment(n_steps=8, epochs=15, n_train=600, n_eval=600,
                         cost_grid=(0.0, 0.002), seed=0)
    rows = res["rows"]
    assert len(rows) == 2 * 4  # 2 costs x 4 strategies
    assert {r["strategy"] for r in rows} == {"deep", "delta", "banded_delta", "no_hedge"}
    assert all("cvar95" in r for r in rows)


def test_deep_not_worse_than_no_hedge_at_cost():
    res = run_experiment(n_steps=10, epochs=80, n_train=2000, n_eval=2000,
                         cost_grid=(0.002,), focus_cost=0.002, seed=0)
    by = {r["strategy"]: r["cvar95"] for r in res["rows"]}
    assert by["deep"] < by["no_hedge"]  # hedging at all beats not hedging


def test_cli_quick_writes_outputs(tmp_path):
    main(["--quick", "--out", str(tmp_path)])
    for f in ("summary.csv", "summary.md", "cvar_vs_cost.png", "pnl_hist.png",
              "training.png", "example_hedge.png"):
        assert (tmp_path / f).exists(), f
