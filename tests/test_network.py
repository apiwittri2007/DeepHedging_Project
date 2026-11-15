import autograd.numpy as anp
import numpy as np
from autograd import grad

from deephedge.models.network import init_params, make_policy, mlp_forward


def test_param_shapes():
    params = init_params([3, 8, 8, 1], seed=0)
    shapes = [(W.shape, b.shape) for W, b in params]
    assert shapes == [((3, 8), (8,)), ((8, 8), (8,)), ((8, 1), (1,))]


def test_forward_shape_and_finite():
    params = init_params([3, 8, 1], seed=1)
    feats = np.random.default_rng(0).normal(size=(20, 3))
    out = mlp_forward(params, feats)
    assert out.shape == (20,)
    assert np.isfinite(np.asarray(out)).all()


def test_forward_deterministic():
    params = init_params([3, 8, 1], seed=2)
    feats = np.ones((5, 3))
    assert np.allclose(np.asarray(mlp_forward(params, feats)), np.asarray(mlp_forward(params, feats)))


def test_forward_differentiable_wrt_params():
    params = init_params([3, 4, 1], seed=3)
    feats = np.ones((6, 3))
    g = grad(lambda p: anp.sum(mlp_forward(p, feats)))(params)
    assert len(g) == len(params)
    assert g[0][0].shape == params[0][0].shape


def test_make_policy_matches_forward():
    params = init_params([3, 4, 1], seed=4)
    feats = np.ones((5, 3))
    pol = make_policy(params)
    assert np.allclose(np.asarray(pol(feats)), np.asarray(mlp_forward(params, feats)))
