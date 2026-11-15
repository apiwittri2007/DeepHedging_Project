from pathlib import Path

from deephedge.experiments.run_training_data import main
from deephedge.plots.training_data_charts import CHARTS


def test_cli_quick_writes_tables_and_charts(tmp_path):
    out, figs = tmp_path / "res", tmp_path / "fig"
    main(["--quick", "--n-jobs", "1", "--out", str(out), "--figures", str(figs)])
    for name in ("summary.md", "summary.csv", "significance.csv", "results.pkl"):
        assert (out / name).exists()
    for name in CHARTS:
        assert (figs / name).stat().st_size > 10_000
    # redraw from the saved results without retraining
    main(["--from-results", "--out", str(out), "--figures", str(figs)])


def test_merge_keeps_existing_rows_and_adds_new_arm():
    from deephedge.experiments.run_training_data import merge_results
    base = {"summary": [{"arm": "A1_gbm", "v": 1}], "significance": [], "by_year": [],
            "realism": [{"source": "real", "v": 1}]}
    new = {"summary": [{"arm": "A1_gbm", "v": 2}, {"arm": "A2b_garch_nu4", "v": 3}], "significance": [],
           "by_year": [], "realism": [{"source": "real", "v": 2}, {"source": "A2b_garch_nu4", "v": 3}]}
    m = merge_results(base, new)
    assert m["summary"] == [{"arm": "A1_gbm", "v": 1}, {"arm": "A2b_garch_nu4", "v": 3}]
    assert [r["v"] for r in m["realism"]] == [1, 3]
