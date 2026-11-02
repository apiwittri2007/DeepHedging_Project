import autograd.numpy as anp
import numpy as np
from autograd import grad

from deephedge.costs import proportional_cost
from deephedge.gbm import simulate_gbm_paths
from deephedge.hedge import simulate_hedged_pnl


def test_proportional_cost():
    assert proportional_cost(1000.0, 0.001) == 1.0


def const_policy(theta):
    return lambda feats: anp.ones(feats.shape[0]) * theta


def no_hedge(feats):
    return anp.zeros(feats.shape[0])


def test_no_hedge_pnl_is_premium_minus_payoff():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 500, seed=1)
    pnl = simulate_hedged_pnl(paths, no_hedge, K=100, r=0.0, sigma=0.2, T=1.0, cost_rate=0.0, premium=8.0)
    payoff = np.maximum(paths[:, -1] - 100, 0.0)
    assert np.allclose(np.asarray(pnl), 8.0 - payoff)


def test_constant_holding_closed_form():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 400, seed=2)
    pnl = np.asarray(simulate_hedged_pnl(paths, const_policy(0.5), 100, 0.0, 0.2, 1.0, 0.0, premium=8.0))
    payoff = np.maximum(paths[:, -1] - 100, 0.0)
    expected = 8.0 + 0.5 * (paths[:, -1] - paths[:, 0]) - payoff
    assert np.allclose(pnl, expected)


def test_costs_reduce_pnl_for_trading_policy():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 50, 2000, seed=3)

    def wiggle(feats):  # a policy that trades every step
        return 0.5 + 0.1 * anp.sign(feats[:, 1])

    free = np.asarray(simulate_hedged_pnl(paths, wiggle, 100, 0.0, 0.2, 1.0, 0.0, premium=8.0)).mean()
    costly = np.asarray(simulate_hedged_pnl(paths, wiggle, 100, 0.0, 0.2, 1.0, 0.002, premium=8.0)).mean()
    assert costly < free


def test_engine_is_autograd_differentiable():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 10, 300, seed=4)

    def mean_pnl(theta):
        return anp.mean(simulate_hedged_pnl(paths, const_policy(theta), 100, 0.0, 0.2, 1.0, 0.0, premium=8.0))

    g = grad(mean_pnl)(0.5)
    expected = (paths[:, -1] - paths[:, 0]).mean()  # d/dtheta of closed form
    assert np.isclose(g, expected, rtol=1e-6)
