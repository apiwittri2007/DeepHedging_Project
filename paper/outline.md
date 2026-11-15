# Outline — Which training data makes deep hedging work on real Hong Kong stocks?

Numbers below come from `results/training_data/summary.md` (run of 2026-11-15, 990 training jobs).

## 1. Introduction
- Deep hedging (Buehler et al. 2019) is almost always trained on simulated paths. Practitioners ask
  whether simulated, realistic-simulated or real data trains a better hedger.
- Evidence so far: Mikkilä & Kanniainen (2023) train on real S&P 500 option data; Ruf & Wang (2022)
  find a linear regression matches neural networks once the evaluation split is done right.
- Gap: a controlled comparison of training-data sources on one untouched real test set, with
  walk-forward retraining, several seeds and a clustered significance test — in an emerging,
  high-cost market (Hong Kong tech stocks, ≥ 10 bps stamp duty).
- **Contribution:** (i) the comparison; (ii) showing that a single-split, single-seed result
  overstates the deep hedger's edge; (iii) a realism-vs-performance analysis of generators.

## 2. Data
- 29 Hang Seng Tech stocks, daily adjusted closes 2000–2026 (free, Yahoo via public GitHub mirror).
- 1-month ATM calls, trailing 63-day vol for premium and delta; data-error filter (|log r| > 0.5).
- Test set: 2,650 non-overlapping windows 2016–2026; folds per year; training data ends before the year.
- Stylised facts: vol-scaled excess kurtosis 6.7, Hill index 3.5 (pre-year averages).

## 3. Method
- Policy MLP 4-32-32-1 on [ttm, log-moneyness, holding, BS delta]; CVaR₉₅ via Rockafellar–Uryasev.
- Arms A0–A5 and benchmarks (BS delta, band, no hedge, L linear correction); fixed settings, never tuned on 2016+.
- Statistics: seed-averaged CVaR; paired bootstrap clustered by calendar month; 6 arms × 6 costs —
  report as exploratory beyond the primary 15 bps comparison.

## 4. Results (pre-registered hypotheses, primary cost 15 bps)
| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 real-informed arms (A2–A5) beat A1 | **Not supported** | vs A1: A3 −0.03, A5 −0.03, A4 +0.10, all CIs include 0; A2 +2.11 (worse) |
| H2 best deep arm beats linear benchmark L | **Not supported** | L has the lowest CVaR (6.21); best deep arm A5 6.23; L is the only method significantly better than delta at 0–25 bps (15 bps: −0.14 [−0.24, −0.05]) |
| H3 deep-hedging gain grows with cost | **Supported** | at 50 bps A0 −0.27, A1 −0.31, A5 −0.38 vs delta, CIs below 0; at 15 bps none significant |

Further findings:
- The earlier single-split, single-seed result (deep − delta = −0.25 at 15 bps, significant) shrinks
  to −0.05 [−0.20, +0.08] under walk-forward and 3 seeds. Seed sd (0.03–0.11) is the size of the effect.
- Realism: block bootstrap matches real tails and clustering best (kurtosis 7.7, Hill 3.4, ACF 0.04);
  GBM has none (0.1, 6.9, −0.01); GARCH-t overshoots badly (136, 3.2, 0.11) and makes training unstable.
  Realism did not translate into better hedging beyond noise.
- Stability: gaps vary in sign year to year for every deep arm; L is negative in 10 of 11 years.
- Data quantity: A4's gap shows no trend with 22k → 72k real training windows.

Figures: `figures/training_data/td_*.png`.

## 5. Robustness / to do before submission
- GARCH-t variant with a tail floor (decision pending)
- More seeds (5–10) for the primary comparison; Holm correction across arms.
- Hull & White (2017) minimum-variance delta as a second linear benchmark (reference to verify).

## 6. Conclusion
Under an honest protocol, the choice of training data matters less than the evaluation protocol:
deep hedgers beat delta only when costs are high, and a three-parameter linear correction is as good
or better at realistic costs.
