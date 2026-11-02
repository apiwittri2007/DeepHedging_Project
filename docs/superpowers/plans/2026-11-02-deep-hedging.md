# Deep Hedging Implementation Plan (as built)

**Spec:** `docs/superpowers/specs/2026-11-02-deep-hedging-design.md`
Executed inline with TDD; one review pass. Tasks, each ending green:

1. **GBM + Black–Scholes** (`gbm.py`) — seeded path simulation; call price/delta with tau=0 handling.
2. **Hedging engine + costs** (`hedge.py`, `costs.py`) — causal, self-financing, autograd-differentiable P&L; proportional costs. Tests include a gradient check and the zero-cost no-hedge closed form.
3. **Delta policies** (`delta.py`) — BS delta and no-trade band; zero-cost risk-reduction sanity test.
4. **MLP + metrics** (`network.py`, `metrics.py`) — autograd MLP; CVaR, Rockafellar–Uryasev objective, summary.
5. **Training** (`train.py`) — R–U CVaR objective, autograd + Adam; test that tail risk falls out-of-sample.
6. **Experiment + CLI + docs** (`experiment.py`, `__main__.py`) — train across a cost grid vs benchmarks; real table + 4 plots; README + interview notes.

**Review-pass change:** added the BS delta as a 4th policy feature so the network learns a
cost-aware correction to delta (before this it underperformed delta at all cost levels);
after, it matches delta at zero cost and beats it as costs rise. Fixed a 3-feature bug in the
example-path helper surfaced by that change.
