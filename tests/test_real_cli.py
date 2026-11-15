import numpy as np
import pandas as pd

from deephedge.experiments.run_real import main
from deephedge.plots.real_charts import CHARTS


def test_real_cli_quick_writes_charts(tmp_path):
    data = tmp_path / "hk"
    data.mkdir()
    for i in range(3):
        rng = np.random.default_rng(i)
        n = 2600
        p = 40 * np.exp(np.cumsum(rng.normal(0, 0.025, n)))
        d = pd.bdate_range("2012-01-02", periods=n)
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Close": p}).to_csv(data / f"S{i}.HK.csv", index=False)
    out, figs = tmp_path / "out", tmp_path / "figs"
    main(["--data", str(data), "--out", str(out), "--figures", str(figs), "--quick"])
    for name in ("real_summary.csv", "real_summary.md"):
        assert (out / name).exists(), name
    for name in CHARTS:
        assert (figs / name).exists(), name
