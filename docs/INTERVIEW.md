# Interview notes: deep hedging vs delta hedging on real Hong Kong stocks

## 30-second summary
> "I tested whether a neural-network hedger beats Black–Scholes delta hedging on real Hong Kong
> tech stocks once trading costs are real, and which training data makes it best. The key was the
> protocol: I retrained every model each year using only past data, ran ten seeds, and corrected for
> the sixty comparisons I made. The honest answer: at realistic HK costs, a simple three-parameter
> correction to delta is the only thing that beats delta after correction, about HK$1,400 less tail
> loss per million of notional, though at 15 bps that is borderline. The neural network only pays off when trading is expensive. A result that
> looked significant in my first, single-split version did not survive that protocol."

## Why this isn't a toy "neural net beats the market" project
- **Falsifiable benchmark.** In a GBM world delta is optimal at zero cost, and the network does not
  beat it there. It wins exactly where theory says it can: when costs are high.
- **I reported my own result failing.** The pilot showed −4% tail risk at 15 bps, significant. With
  walk-forward retraining, 10 seeds and multiple-testing correction, it was not robust. I kept
  the pilot in the repo, labelled as superseded.
- **Pre-registration.** Hypotheses, arms and the primary cost were committed before the results existed.

## The hardest parts (what I actually did)
1. **Differentiating CVaR through a hedging path.** P&L comes from 21 sequential decisions. I use
   the Rockafellar–Uryasev form of CVaR, with the VaR level learned jointly, so `autograd` can
   backpropagate through the whole path.
2. **A walk-forward, leak-free protocol.** For each test year every training source uses only prices
   before 1 January, and real training windows must also *end* before it. Tests check this.
3. **Six training generators under one protocol.** I fitted GARCH(1,1)-t by maximum likelihood
   (hand-written, no `arch` package). I also built a stationary block bootstrap, and GBM with the
   trader estimating vol exactly as on real data.
4. **Joint inference.** One month-clustered bootstrap is shared by all 60 tests, which gives Holm and
   Romano–Wolf adjusted p-values. An exact weighted-CVaR identity means no resampled data has to be built.

## Likely questions
- **"Why CVaR, not variance?"** Desks care about tail losses, and variance penalises upside too.
- **"Why does the linear rule win at low cost?"** It needs three parameters, so its estimate is
  stable across years. The network's extra flexibility only pays when there is a lot of cost to save.
- **"Why didn't realistic training data help?"** The block bootstrap reproduces real fat tails but
  trained no better a hedger. What matters is the cost/turnover trade-off the network learns, and
  plain GBM teaches that too. A 2026 paper (Hashimoto et al.) finds the same thing: generator
  realism is not what drives hedging performance.
- **"Is the effect economically meaningful?"** About 2.3% of the option premium at 15 bps
  (HK$1,400 per HK$1m notional). At 50 bps deep hedging saves 5–6%.
- **"Biggest limitation?"** Synthetic option prices. With traded options there is a spot–implied-vol
  co-movement that the Hull–White correction exploits. Replicating on traded options is the next step.
- **"How do you know it's not noise?"** Ten seeds; resampling whole months, because stocks in the same
  month share shocks; and a Romano–Wolf correction over 60 tests. Only the linear correction, and the
  deep hedger at 50 bps, survive. I'm upfront that the 15 bps result is borderline: with quarterly
  clusters its adjusted p is 0.06, while at 25–50 bps it holds under every cluster choice.

## Data story
Yahoo and HKEX weren't reachable from my build machine. I found a free, daily-refreshed GitHub mirror
of Yahoo prices (GPL-3.0), kept only `Date, Close`, documented the source in a NOTICE, and wrote a
script to regenerate it.
