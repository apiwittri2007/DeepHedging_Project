# Real Hong Kong Data Out-of-Sample Test — Design

Date: 2026-11-03. Status: approved in chat ("use real data stocks … find it for free"; autopilot).

## Problem statement

v1 trains and tests the deep hedger on simulated GBM paths, where Black–Scholes is the
true model. Real prices are not GBM: fat tails, jumps, volatility clustering, and the vol
you price with is not the vol you realise. **Does a hedger trained on simulation survive
real Hong Kong stock paths, and does its cost advantage over delta hedging persist?**

## Data (free, real)

`zjplab/US-Trading-Data` (GitHub, GPL-3.0, auto-updated daily from Yahoo Finance via
yfinance): daily split/dividend-adjusted prices for ~30 Hang Seng Tech constituents
(Tencent 0700, Alibaba 9988, Meituan 3690, Xiaomi 1810, SMIC 0981, Lenovo 0992, …),
up to 2000–2026. Yahoo/HKEX are not reachable from the build machine; GitHub is. We store
`Date, Close` per ticker in `data/hk/` with a NOTICE (source + licence) and a script that
regenerates it.

## Experiment

- **Contract:** sell a 1-month (21 trading days) at-the-money European call; rebalance
  daily (21 steps); T = 21/252.
- **Windows:** per stock, non-overlapping 22-price windows, normalised to S₀ = 100, K = 100.
- **Trader's vol:** trailing 63-day realised vol computed from returns strictly *before* the
  window starts (no look-ahead); used for the premium, the delta and the delta feature.
- **Split:** windows starting before 2016-01-01 = calibration; 2016+ = out-of-sample test.
- **Deep hedger training:** simulated GBM paths, each with σ drawn from the calibration
  windows' trailing-vol distribution; same 4 features and CVaR objective as v1.
- **Strategies evaluated on real test windows:** deep hedger, BS delta (trailing vol),
  delta + no-trade band, no hedge; costs 0/5/10/15/25/50 bps per side (15 ≈ realistic HK:
  10 bps stamp duty + fees + half-spread).
- **Data hygiene:** drop windows with missing prices or any |daily log return| > 0.5
  (data errors); report counts.

## Analyses / charts

1. Stylized facts: pooled HK daily returns vs normal (excess kurtosis).
2. Sample real windows vs GBM.
3. Real test CVaR vs cost per strategy.
4. Real P&L histograms at 15 bps.
5. Vol misspecification: delta-hedge P&L vs (trailing σ − realised σ in window, ex-post).
6. Per-period CVaR (COVID 2020-02..06, tech crackdown 2021-02..2022-10, 2023+, all test).

## Correctness rules

- Trailing vol uses only returns dated before the window's first price.
- Realised in-window vol is used **only** for after-the-fact analysis (chart 5), never by a policy.
- Engine/GBM accept per-path σ; scalar behaviour unchanged (all v1 tests pass).

## Out of scope / limitations to state

Survivorship (today's HS Tech members), adjusted prices include dividends, single-stock
hedging (no index futures), Yahoo data quality, overlapping-in-time windows across stocks.
