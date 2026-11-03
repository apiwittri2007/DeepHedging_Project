# Interview notes — Deep hedging vs delta hedging under costs

## 30-second summary

> "I tested whether a neural-network hedger beats Black–Scholes delta hedging once
> transaction costs are real. On simulated GBM paths I train a deep hedger to minimise
> the 95% CVaR of the terminal P&L of a short call, under proportional costs, and compare
> it to delta hedging. The honest finding: at zero cost it only matches delta — it can't
> beat the optimal hedge — but as costs rise it wins by learning to trade less; at 50 bps
> it cuts tail risk ~9% and loses less on average. I built it in NumPy with autograd, no
> deep-learning framework, 30 tests."

## Why it's not a toy "NN beats the market" project

- The benchmark is **provably optimal at zero cost** (BS delta in a GBM world), so the
  comparison is falsifiable — and indeed the NN does *not* beat it at zero cost. A project
  where the NN wins everywhere would be a red flag.
- The win appears exactly where theory says it should (high costs), for a reason I can
  explain (trade less), and I show the regime where it's a tie.

## The three hardest parts (what I actually did)

**1. Differentiating risk through a simulated hedging path.**
The loss is the 95% CVaR of terminal P&L, and P&L is the result of 20 sequential hedging
decisions each depending on the network. I use the **Rockafellar–Uryasev** form of CVaR
(a minimisation over a VaR level `w`, learned jointly) so the objective is differentiable,
and let `autograd` backpropagate through the whole path. Talking point: "CVaR isn't smooth
if you compute it by sorting; the R–U trick makes it a clean minimisation you can take
gradients of."

**2. A causal, self-financing hedging engine.**
The engine accrues cash with proportional costs and only ever feeds the policy information
available at that step — so there's no look-ahead by construction. It's written in
`autograd.numpy` so gradients reach the network, while prices are constants. A unit test
checks the engine is differentiable (gradient of a constant-holding policy matches the
closed form), and another checks delta hedging drives risk → 0 at zero cost.

**3. The model-delta-as-feature design.**
Giving the network the BS delta as an input feature lets it learn a *correction* to delta
rather than relearn hedging from scratch. That's what makes it match delta at zero cost and
improve on it under costs — without it, a small net underperforms delta everywhere (I saw
this before adding the feature).

## Questions I can answer

- "Why CVaR not variance?" → tail risk is what desks care about; variance penalises upside too.
- "Why does it only win at high cost?" → at zero cost delta is optimal; the NN's edge is
  purely cost-saving (fewer trades), which only matters when trading is expensive.
- "How do you know there's no look-ahead?" → the engine passes the policy only step-k state;
  tested.
- "Biggest limitation?" → GBM world and a single option; Heston/rough-vol and portfolios next.

## What I'd do next

Heston / rough-vol paths; a book of options; recurrent policy (holding history);
no-transaction-band architecture (Imaki et al.); dispersion across seeds; compare the CVaR
objective to a mean-variance / entropic utility objective.

---

## Part 2 — Real Hong Kong data (the follow-up question every interviewer asks)

> "Fine in simulation — but does it work on real data?"

**30-second answer.** "I took the hedger trained only on simulated GBM paths and ran it
out-of-sample on 2,650 real one-month windows from 29 Hang Seng Tech stocks, 2016–2026.
Two findings. First, real tail risk is about twice what GBM predicts: returns have excess
kurtosis of about 7, and hedging P&L is mostly a bet on realised versus assumed vol, with
correlation 0.87. Second, the deep hedger's cost advantage survives: at a realistic 15 bps
it cuts CVaR by about 4%, and the month-clustered bootstrap CI is [−0.35, −0.14], so it's
significant. At 0–5 bps delta still wins, which is what theory predicts."

**Likely follow-ups.**
- *"How did you avoid look-ahead on real data?"* → the trader's vol is the trailing 63-day
  vol from returns up to the window start; a test shocks prices inside a window and checks
  the trailing vol doesn't move. In-window realised vol is used only in the after-the-fact
  analysis chart.
- *"Did you train on real data?"* → no, deliberately. Pre-2016 windows only shape the
  distribution of vols for the *simulated* training paths, and everything is tested on 2016+.
  That's the hardest version of the test: the model never saw a jump.
- *"Is −4% just noise?"* → paired bootstrap over windows, resampling whole calendar months,
  because stocks in the same month share shocks. The CI excludes zero at ≥ 15 bps and
  includes zero at 0–10 bps.
- *"Why is real CVaR double?"* → vol misspecification and jumps; no hedging rule can remove
  the vol gap. The network's edge is only on the controllable part: trading cost.
- *"Where does it fail?"* → COVID crash (n = 84, CVaR rests on ~4 tail windows), and low
  costs. Limitations: survivorship (today's members), stock rather than futures hedging,
  trailing rather than implied vol.

**Data story.** Yahoo/HKEX weren't reachable from my build machine, so I found a free,
daily-refreshed GitHub dataset sourced from Yahoo via yfinance (GPL-3.0). I kept only
`Date, Close`, documented provenance in a NOTICE, and wrote a script to regenerate it.
