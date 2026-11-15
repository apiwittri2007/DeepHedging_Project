from deephedge.experiments.run_sim import main
from deephedge.experiments.sim_study import run_experiment


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


def test_cli_quick_writes_tables_and_figures(tmp_path):
    out, figs = tmp_path / "results", tmp_path / "figures"
    main(["--quick", "--out", str(out), "--figures", str(figs)])
    for f in ("summary.csv", "summary.md"):
        assert (out / f).exists(), f
    for f in ("cvar_vs_cost.png", "pnl_hist.png", "training.png", "example_hedge.png"):
        assert (figs / f).exists(), f


def test_cli_config_overrides(tmp_path):
    import json
    cfg = tmp_path / "sim.json"
    cfg.write_text(json.dumps({"n_steps": 6, "epochs": 5, "n_train": 300, "n_eval": 300,
                               "cost_grid": [0.002], "focus_cost": 0.002}))
    res = main(["--config", str(cfg), "--out", str(tmp_path / "r"), "--figures", str(tmp_path / "f")])
    assert {r["cost"] for r in res["rows"]} == {0.002}
