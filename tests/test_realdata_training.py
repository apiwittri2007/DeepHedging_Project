import numpy as np
import pandas as pd

from deephedge.data.realdata import build_windows, returns_before


def series(n, seed=0, vol=0.02, start="2014-01-01"):
    rng = np.random.default_rng(seed)
    p = 50 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    return pd.Series(p, index=pd.bdate_range(start, periods=n))


def test_step1_overlapping_count():
    s = series(63 + 21 + 50)
    w = build_windows({"A": s}, step=1)
    assert len(w) == 50                     # one window per possible start day (134 prices)
    assert len(build_windows({"A": s})) == 3  # default step = horizon


def test_end_field_is_last_price_date():
    s = series(63 + 21 * 3)
    w = build_windows({"A": s})
    for st, en in zip(w.start, w.end):
        i = s.index.get_loc(pd.Timestamp(st))
        assert pd.Timestamp(en) == s.index[i + 21]


def test_end_before_filters_by_last_price():
    s = series(63 + 21 * 20)
    cut = "2015-06-01"
    w = build_windows({"A": s}, step=1, end_before=cut)
    assert len(w) > 0
    assert (w.end < np.datetime64(cut)).all()
    full = build_windows({"A": s}, step=1)
    assert len(w) == int((full.end < np.datetime64(cut)).sum())


def test_returns_before_excludes_date_and_errors():
    s = series(300)
    cut = s.index[200]
    s2 = s.copy()
    s2.iloc[50] *= 3.0          # data error: two returns of |log| > 0.5
    s2.iloc[120] = np.nan       # missing price
    r = returns_before({"A": s2}, str(cut.date()))["A"]
    lr = np.log(s2.values[1:200] / s2.values[:199])
    expected = lr[np.isfinite(lr) & (np.abs(lr) <= 0.5)]
    assert np.allclose(r, expected)
