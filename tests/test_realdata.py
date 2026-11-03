import numpy as np
import pandas as pd
import pytest

from deephedge.realdata import build_windows, load_closes


def series(n, seed=0, vol=0.02, start="2010-01-04"):
    rng = np.random.default_rng(seed)
    p = 50 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(p, index=pd.bdate_range(start, periods=n))


def test_window_shape_and_normalisation():
    w = build_windows({"AAA": series(63 + 22 * 5)}, horizon=21, vol_lookback=63)
    assert w.paths.shape[1] == 22
    assert np.allclose(w.paths[:, 0], 100.0)
    assert len(w.paths) == len(w.trailing_vol) == len(w.realised_vol) == len(w.start) == len(w.ticker)
    assert (w.ticker == "AAA").all()


def test_non_overlapping():
    s = series(63 + 22 * 6)
    w = build_windows({"A": s}, horizon=21, vol_lookback=63)
    gaps = np.diff(w.start.astype("datetime64[D]").astype(int))
    assert (gaps > 0).all()
    idx = [s.index.get_loc(pd.Timestamp(d)) for d in w.start]
    assert np.all(np.diff(idx) == 21)


def test_trailing_vol_excludes_window():
    s = series(63 + 22 * 3, seed=1)
    base = build_windows({"A": s})
    shocked = s.copy()
    j = s.index.get_loc(pd.Timestamp(base.start[0]))
    shocked.iloc[j + 5:] *= 1.3          # a jump inside the first window
    after = build_windows({"A": shocked})
    assert np.isclose(after.trailing_vol[0], base.trailing_vol[0])
    assert after.realised_vol[0] > base.realised_vol[0]


def test_trailing_vol_annualised():
    w = build_windows({"A": series(63 + 22 * 40, seed=2, vol=0.02)})
    assert np.isclose(np.median(w.trailing_vol), 0.02 * np.sqrt(252), rtol=0.1)


def test_short_series_gives_no_windows():
    w = build_windows({"A": series(70)})
    assert len(w.paths) == 0 and w.paths.shape[1] == 22


def test_nan_window_dropped():
    s = series(63 + 22 * 4, seed=3)
    clean = build_windows({"A": s})
    s2 = s.copy()
    j = s.index.get_loc(pd.Timestamp(clean.start[1]))
    s2.iloc[j + 3] = np.nan
    w = build_windows({"A": s2})
    assert len(w.paths) == len(clean.paths) - 1
    assert w.n_dropped >= 1


def test_extreme_return_window_dropped():
    s = series(63 + 22 * 4, seed=4)
    clean = build_windows({"A": s})
    j = s.index.get_loc(pd.Timestamp(clean.start[-1]))
    s2 = s.copy()
    s2.iloc[j + 10:] *= 3.0             # +110% log-return in a day: data error
    w = build_windows({"A": s2}, max_abs_logret=0.5)
    assert len(w.paths) == len(clean.paths) - 1


def test_load_closes_roundtrip(tmp_path):
    s = series(30)
    pd.DataFrame({"Date": s.index.strftime("%Y-%m-%d"), "Close": s.values}).to_csv(tmp_path / "0700.HK.csv", index=False)
    closes = load_closes(tmp_path)
    assert list(closes) == ["0700.HK"]
    assert np.allclose(closes["0700.HK"].values, s.values)


def test_subset():
    w = build_windows({"A": series(63 + 22 * 5), "B": series(63 + 22 * 5, seed=9)})
    sub = w.subset(w.ticker == "B")
    assert (sub.ticker == "B").all() and len(sub.paths) == (w.ticker == "B").sum()
