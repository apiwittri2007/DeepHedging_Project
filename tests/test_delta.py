import autograd.numpy as anp
import numpy as np

from deephedge.hedging.delta import banded_delta_policy, delta_policy
from deephedge.sim.gbm import simulate_gbm_paths
from deephedge.hedging.hedge import simulate_hedged_pnl


def feats(ttm, logm, shares):
    return anp.array([[ttm, logm, shares]])


def test_delta_policy_atm_value():
    p = delta_policy(K=100, r=0.0, sigma=0.2)
    assert np.isclose(p(feats(1.0, 0.0, 0.0))[0], 0.5398, atol=1e-3)


def test_delta_policy_deep_itm_and_otm():
    p = delta_policy(100, 0.0, 0.2)
    assert p(feats(0.5, np.log(1000 / 100), 0.0))[0] > 0.999
    assert p(feats(0.5, np.log(10 / 100), 0.0))[0] < 0.001


def test_banded_policy_holds_inside_band_moves_outside():
    p = banded_delta_policy(100, 0.0, 0.2, band=0.1)
    # ATM delta ~0.54; current shares 0.5 -> gap 0.04 < band -> keep 0.5
    assert np.isclose(p(feats(1.0, 0.0, 0.5))[0], 0.5)
    # current shares 0.0 -> gap 0.54 > band -> move to delta
    assert np.isclose(p(feats(1.0, 0.0, 0.0))[0], 0.5398, atol=1e-3)


def test_delta_hedging_reduces_risk_zero_cost():
    paths = simulate_gbm_paths(100, 0.0, 0.2, 1.0, 100, 5000, seed=5)  # risk-neutral r=0
    pol = delta_policy(100, 0.0, 0.2)
    hedged = np.asarray(simulate_hedged_pnl(paths, pol, 100, 0.0, 0.2, 1.0, 0.0))
    unhedged = np.asarray(
        simulate_hedged_pnl(paths, lambda f: anp.zeros(f.shape[0]), 100, 0.0, 0.2, 1.0, 0.0)
    )
    assert hedged.std() < 0.35 * unhedged.std()   # delta hedging cuts risk sharply
    assert abs(hedged.mean()) < 0.1                # ~self-financing in the BS limit
