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
