# Hong Kong price data — source and licence

- **What:** daily closing prices (split- and dividend-adjusted) for 29 Hang Seng Tech
  Index constituents, e.g. Tencent `0700.HK`, Alibaba `9988.HK`, Meituan `3690.HK`,
  Xiaomi `1810.HK`, SMIC `0981.HK`, Lenovo `0992.HK`. History from 2000 (oldest
  listings) to 2026-11-03.
- **Where from:** [`zjplab/US-Trading-Data`](https://github.com/zjplab/US-Trading-Data),
  commit `933159c` (2026-11-03), which downloads them daily from Yahoo Finance via the
  `yfinance` package.
- **Licence:** that repository is GPL-3.0; these files are a column subset (`Date`,
  `Close`) of its `data/HangSengTech/*.csv` and remain under GPL-3.0. Underlying market
  data © its providers; used here for non-commercial research/education.
- **Regenerate / refresh:** see `scripts/fetch_hk_data.py`.
- **Caveats:** Yahoo Finance data can contain errors (the pipeline drops windows with any
  |daily log return| > 0.5); the stock list is *today's* index members (survivorship bias).
