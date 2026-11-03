import numpy as np
import pandas as pd

from deephedge.real import CHARTS, main


def test_real_cli_quick_writes_charts(tmp_path):
    data = tmp_path / "hk"
    data.mkdir()
    for i in range(3):
        rng = np.random.default_rng(i)
        n = 2600
        p = 40 * np.exp(np.cumsum(rng.normal(0, 0.025, n)))
        d = pd.bdate_range("2012-01-02", periods=n)
        pd.DataFrame({"Date": d.strftime("%Y-%m-%d"), "Close": p}).to_csv(data / f"S{i}.HK.csv", index=False)
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--quick"])
    for name in list(CHARTS) + ["real_summary.csv", "real_summary.md"]:
        assert (out / name).exists(), name
