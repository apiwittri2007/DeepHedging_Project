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
