# Deep Hedging vs Delta Hedging under Transaction Costs — Design

Date: 2026-11-02. Status: approved in chat ("Deep hedging vs delta hedging under costs").

## Problem statement

Black–Scholes delta hedging is optimal only in a frictionless, continuous-time world.
With discrete rebalancing and proportional transaction costs, constantly trading to the
delta bleeds money. **Deep hedging** (Bühler et al., 2019) instead trains a neural network
to choose hedge positions that directly minimise a risk measure of the terminal P&L,
learning to trade less when costs outweigh risk reduction.

**Question:** on simulated GBM paths, does a trained deep hedger achieve lower tail risk
(CVaR₉₅ of P&L) than Black–Scholes delta hedging once proportional costs are present, and
in which regimes (cost level, rebalancing frequency) is the gain real?

## Why this is a strong, honest project

- Monte Carlo is the core: everything is simulated, so results are reproducible here with
  no data-access problem.
- GBM world → BS delta is the provably-optimal zero-cost hedge, so the benchmark is fair and
  a correctness check exists (delta-hedge error → 0 as steps → ∞ with zero cost).
- Honest framing: report where deep hedging does *not* help, not just where it wins.

## Design

Pure simulation; `autograd.numpy` where gradients must flow (PyTorch's host is blocked here).

1. `gbm.py` — simulate GBM price paths from fixed Brownian increments (seeded); Black–Scholes
   call price and delta (plain numpy/scipy) for benchmark and sanity checks.
2. `costs.py` — `proportional_cost(trade_notional, rate)`; rate in decimal (e.g. 0.001 = 10 bps).
3. `hedge.py` — the engine: given a policy (callable mapping per-step state → target holding)
   and price paths, simulate holdings, cash, option payoff and costs; return terminal P&L per
   path. Causal by construction (policy sees only current state). Written in autograd.numpy so
   gradients reach a network policy.
4. `delta.py` — BS-delta policy and a no-trade-band delta policy (don't trade if |Δ gap| < band).
5. `network.py` — small MLP (autograd params); state (ttm, log-moneyness, holding) → holding.
6. `metrics.py` — `cvar(pnl, alpha)`, `summary(pnl)` (mean, std, CVaR, VaR).
7. `train.py` — minimise the Rockafellar–Uryasev CVaR objective (with a learnable VaR level w)
   over minibatches of simulated paths via autograd + Adam; return trained policy + loss curve.
8. `experiment.py` + `__main__.py` — train the deep hedger, evaluate vs BS-delta, banded-delta
   and no-hedge across a grid of cost levels; write results table + plots (P&L histograms,
   CVaR-vs-cost curve, example hedge path).

## Correctness rules

- No look-ahead: the policy at step k receives only (time, current price/holding) at k.
- Zero-cost sanity: BS-delta terminal P&L std shrinks toward 0 as rebalancing steps increase.
- CVaR is the mean of the worst (1−alpha) tail of P&L (loss side), computed consistently.
- Brownian increments are fixed data per evaluation; only network params are trained.

## Out of scope (stated)

Heston / rough vol (possible extension), American/path-dependent payoffs, real market data,
multi-asset hedging. v1 is GBM + European call + one hedging instrument.
