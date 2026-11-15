"""Real Hong Kong price data -> 1-month option-hedging windows.

Each window is what an option seller would face: 22 daily prices (21 daily
rebalances), normalised so the first price is 100 and the strike is at the
money. The volatility the seller prices and hedges with is the *trailing*
63-day realised vol, using only returns that end on or before the window's
first price — so there is no look-ahead. The realised vol inside the window is
stored separately for after-the-fact analysis only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_TRAILING_VOL = 0.01  # below this the price is stale (no trading), not a real regime


@dataclass
class Windows:
    paths: np.ndarray          # (n, horizon + 1), first price = 100
    trailing_vol: np.ndarray   # (n,) annualised, known at window start
    realised_vol: np.ndarray   # (n,) annualised, inside the window (ex-post only)
    start: np.ndarray          # (n,) datetime64[D] of the first price
    ticker: np.ndarray         # (n,) str
    n_dropped: int = 0

    def subset(self, mask) -> "Windows":
        mask = np.asarray(mask, dtype=bool)
        return Windows(self.paths[mask], self.trailing_vol[mask], self.realised_vol[mask],
                       self.start[mask], self.ticker[mask], self.n_dropped)

    def __len__(self) -> int:
        return len(self.paths)


def load_closes(data_dir: str | Path) -> dict[str, pd.Series]:
    """Read ``<ticker>.csv`` files with columns ``Date, Close`` into price series."""
    closes = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path, parse_dates=["Date"])
        s = pd.Series(df["Close"].to_numpy(dtype=float), index=df["Date"]).sort_index()
        closes[path.stem] = s.where(s > 0)
    return closes


def build_windows(
    closes: dict[str, pd.Series],
    horizon: int = 21,
    vol_lookback: int = 63,
    max_abs_logret: float = 0.5,
) -> Windows:
    """Non-overlapping option windows per stock, with no-look-ahead trailing vol.

    A window is dropped if it (or its trailing-vol history) contains a missing
    price, a daily |log return| above ``max_abs_logret`` (a data error rather
    than a market move), or a stale trailing vol below 1%.
    """
    paths, tvol, rvol, start, tick = [], [], [], [], []
    dropped = 0
    ann = np.sqrt(TRADING_DAYS)
    for name, s in closes.items():
        p = s.to_numpy(dtype=float)
        dates = s.index.to_numpy().astype("datetime64[D]")
        lr = np.full(len(p), np.nan)
        lr[1:] = np.log(p[1:] / p[:-1])
        j = vol_lookback
        while j + horizon <= len(p) - 1:
            win = p[j: j + horizon + 1]
            trail = lr[j - vol_lookback + 1: j + 1]      # returns ending on/before p[j]
            inside = lr[j + 1: j + horizon + 1]          # the window's own returns
            valid_trail = trail[~np.isnan(trail)]
            bad = (
                np.isnan(win).any()
                or len(valid_trail) < 0.8 * vol_lookback      # tolerate a few suspension days
                or np.abs(inside).max() > max_abs_logret
                or np.abs(valid_trail).max() > max_abs_logret
            )
            tv = np.nan if bad else valid_trail.std(ddof=1) * ann
            if bad or tv < MIN_TRAILING_VOL:
                dropped += 1
            else:
                paths.append(win / win[0] * 100.0)
                tvol.append(tv)
                rvol.append(inside.std(ddof=1) * ann)
                start.append(dates[j])
                tick.append(name)
            j += horizon
    n = horizon + 1
    return Windows(
        paths=np.array(paths).reshape(-1, n),
        trailing_vol=np.array(tvol, dtype=float),
        realised_vol=np.array(rvol, dtype=float),
        start=np.array(start, dtype="datetime64[D]"),
        ticker=np.array(tick, dtype=str),
        n_dropped=dropped,
    )
