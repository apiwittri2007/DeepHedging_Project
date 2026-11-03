# Project Summary — Deep Hedging under Transaction Costs (Simulation → Real Hong Kong Data)

**Author:** Apiwit · **Stack:** Python, NumPy, autograd, pandas, matplotlib · **Tests:** 53 · **Date:** November 2026

## 1. The question
Black–Scholes delta hedging is optimal only if you can rebalance continuously and for free. In practice you rebalance daily and pay to trade (in Hong Kong, stamp duty alone is 10 bps per side). **Deep hedging** (Bühler, Gonon, Teichmann & Wood, 2019) trains a neural network to choose hedge positions that directly minimise the tail risk of the hedging P&L, including costs. This project asks two things. Does it beat delta hedging in a controlled simulation? Does that edge survive on real stock paths it was never trained on?

## 2. What I built
| Component | What it does |
|---|---|
| Monte Carlo engine | GBM paths (per-path vol); self-financing, causal hedging P&L with proportional costs, written in `autograd.numpy` so the loss is differentiable |
| Deep hedger | MLP: `[time-to-maturity, log-moneyness, holding, BS delta]` → holding; trained by minimising CVaR₉₅ (Rockafellar–Uryasev) with Adam, through the whole simulated path |
| Benchmarks | Black–Scholes delta, delta with a no-trade band, no hedge |
| Real-data pipeline | 29 Hang Seng Tech stocks (2000–2026, Yahoo via a free GitHub dataset) → 3,727 one-month option windows with no-look-ahead trailing vol |
| Statistics | CVaR, paired month-clustered bootstrap confidence intervals, regime breakdowns, stylized-fact tests |

## 3. Results
**Simulation (GBM):** at zero cost the deep hedger only matches delta (CVaR 3.40 vs 3.51), as it must, since delta is optimal there. As costs rise it pulls ahead by trading less: at 50 bps, CVaR 4.68 vs 5.12 (−9%).

**Real HK data, out-of-sample 2016–2026 (2,650 windows), hedger trained only on simulation:**
- Real tail risk is **~2×** what the simulation predicts. Returns have excess kurtosis ≈ 7, and hedging P&L is dominated by the realised-vs-assumed volatility gap (correlation 0.87).
- At a **realistic 15 bps**, deep hedger CVaR₉₅ = **6.10 vs 6.35** for delta, difference −0.25, **95% CI [−0.35, −0.14]** with the month-clustered bootstrap. That is significant, and the gap grows to −0.45 at 50 bps.
- At 0–5 bps delta is still better, and at 10 bps it's a tie. This matches theory: the network's edge is cost management, not a better model.
- By regime it is better over the whole test period, the 2021–22 tech crackdown and 2023 onwards, and worse in the COVID crash (n = 84, noisy).

## 4. What makes it rigorous
- **No look-ahead,** enforced by tests: the hedging engine only shows the policy step-k information, and trailing vol is unchanged by any shock inside the window.
- **Honest benchmark:** in GBM, delta is provably optimal at zero cost, so a network that "wins everywhere" would signal a bug.
- **Out-of-sample by construction:** the hedger never trains on a real path. Pre-2016 data only shapes the simulated vol distribution.
- **Conservative significance:** the bootstrap resamples whole months, so correlated stocks don't count as independent evidence.
- **Reported failures:** the first network lost to delta everywhere. Giving it the model delta as an input fixed that, and the change is documented. Negative regimes (0–5 bps, COVID) are reported, not hidden.

## 5. Limitations
Survivorship bias (today's index members); single-stock hedging rather than index futures; trailing realised vol instead of market implied vol; one training seed per cost; Yahoo data quality.

## 6. Next steps
Heston / rough-volatility training paths; hedging with HSI futures; recurrent policy; dispersion across training seeds; free HKEX index-option implied vols if a source becomes available.

## 7. CV bullet
> Built a deep-hedging engine (NumPy + autograd) that minimises CVaR of option-hedging P&L under transaction costs. Trained only on Monte Carlo paths, it cut 95% CVaR by ~4% vs Black–Scholes delta out-of-sample on 2,650 real Hang Seng Tech option windows (2016–2026) at realistic HK costs, significant under a month-clustered bootstrap. Also showed real tail risk is ~2× the GBM prediction, driven by volatility misspecification. 53 tests.
