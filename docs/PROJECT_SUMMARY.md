# Project Summary: Does Deep Hedging Survive an Honest Evaluation?

**Author:** Apiwit · **Stack:** Python, NumPy, autograd, pandas, SciPy, matplotlib, LaTeX · **Tests:** 87 · **Updated:** 17 November 2026

## 1. The question
Black–Scholes delta hedging is optimal only with free, continuous trading. In Hong Kong you
rebalance daily and pay at least 10 bps stamp duty per side. **Deep hedging** (Buehler et al.,
2019) trains a neural network to choose hedge positions that minimise tail risk (CVaR₉₅) after
costs. This project asks three questions:
- Does deep hedging beat delta on **real** Hong Kong stocks?
- **Which training data** makes the best hedger?
- Does the answer survive an **honest evaluation protocol**?

## 2. What I built
| Component | What it does |
|---|---|
| Hedging engine | Causal, self-financing P&L with proportional costs, differentiable end to end (`autograd`) |
| Deep hedger | MLP `[time to maturity, log-moneyness, holding, BS delta] → holding`, trained on CVaR₉₅ (Rockafellar–Uryasev) |
| Training-data generators | GBM (known or estimated vol), GARCH(1,1)-t fitted by maximum likelihood, stationary block bootstrap, real windows, pre-train + fine-tune |
| Benchmarks | BS delta, no-trade band, no hedge, and two 3-parameter linear corrections (Ruf–Wang-type and the Hull–White minimum-variance form) |
| Real data | 29 Hang Seng Tech stocks 2000–2026 → 2,650 one-month test windows (2016–2026), no look-ahead |
| Protocol | Walk-forward by year (train only on the past), 10 seeds, 3,960 training jobs (4,620 networks) |
| Statistics | Month-clustered paired bootstrap; Holm and Romano–Wolf multiple-testing corrections over 60 tests; results in HK$ and % of premium |
| Reproducibility | One script reproduces everything, pinned environment, per-window P&L as CSV, CI, MIT licence, LaTeX paper draft |

## 3. Results
- **Realistic cost (15 bps):** a 3-parameter **linear correction** to delta is the only method that
  significantly beats delta after correcting for multiple tests: CVaR −0.14 [−0.23, −0.04],
  adjusted p = 0.03, about **HK$1,400 less tail loss per HK$1m notional**. It beats delta in 10 of 11 years.
  The significance is borderline (p = 0.06 with quarterly bootstrap clusters); at 25–50 bps it holds under
  every cluster choice. The best deep hedgers reach the same point estimate but are not significant.
- **High cost (50 bps):** deep hedgers cut tail risk by 0.30–0.37 (5–6% of the premium), 1.7–2× the
  linear correction. The hedger trained on GBM with known vol (A0) stays significant after correction;
  the others are borderline.
- **Protocol matters.** A pilot trained once (one split, one seed) showed a significant −0.25 at
  15 bps. Under yearly retraining it is −0.05 with 3 seeds and −0.14 with 10. That is not significant after
  correction, and seed noise is as large as the effect.
- **Realistic training data does not help.** The block bootstrap matches real fat tails best, but
  trains no better a hedger than plain GBM. GARCH-t overshoots the tails and makes training unstable.

## 4. What makes it rigorous
- **Pre-registered design.** The design (hypotheses, arms, primary cost) was committed before results existed;
  the one post hoc arm is labelled as such.
- **No look-ahead, enforced by tests.** Training data for test year Y uses only prices before 1 Jan Y.
- **Conservative inference.** Resampling whole months (correlated stocks are not independent
  evidence), correcting for 60 comparisons, and checking 2- and 3-month clusters.
- **Negative results reported.** The pilot's headline was overturned, and the repo says so.

## 5. Limitations
- Option prices are synthetic (priced with trailing realised vol), not traded quotes. This is the main one.
- Survivorship bias: the stocks are today's index members.
- Hedging uses the stock, not index futures.

## 6. Next steps
- Replicate on traded options (Deribit BTC options, free API; HKEX stock options if access is granted).
- Post the paper on arXiv with a faculty co-author.
- Submit to an ICAIF 2027 workshop, then *Quantitative Finance* / *Journal of Computational Finance*.

## 7. CV bullets
**Quant / research version**
> Ran a pre-registered walk-forward study (4,620 trained networks, 10 seeds) of deep hedging vs
> Black–Scholes delta on 2,650 real Hang Seng Tech option windows. Showed that a single-split gain does
> not survive multiple seeds and a Romano–Wolf correction. A 3-parameter linear correction cut 95% CVaR by
> 2.3% of premium (HK$1.4k per HK$1m) at realistic 15 bps costs, while deep hedgers only won at 50 bps.
> Python/autograd, 87 tests, LaTeX paper draft.

**IB / asset-management version**
> Analysed the cost of hedging option exposure on 29 Hong Kong tech stocks (2016–2026) under real
> trading costs. Found a simple delta adjustment saves ~HK$1.4k of tail loss per HK$1m notional at
> realistic costs, and identified when more complex (machine-learning) hedging is worth it. Wrote it up
> as a research paper with full statistical testing.
