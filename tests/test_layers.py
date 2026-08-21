import numpy as np
import pytest
import torch

from src.alggeom.polynomial import Polynomial, Variable
from src.layers.polynomial_layer import EarlyStopping, PolynomialLayer
from src.layers.orthopoly_layer import OrthoPolyLayer
from src.layers.rational_layer import RationalPolyLayer
from src.layers.groebner_layer import GroebnerLayer, create_groebner_layer


def test_polynomial_layer_forward_shape():
    layer = PolynomialLayer(input_dim=3, output_dim=2, max_degree=2)
    out = layer(torch.randn(5, 3))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_polynomial_layer_symbolic_forward():
    layer = PolynomialLayer(input_dim=2, output_dim=3, max_degree=2)
    polys = layer.symbolic_forward()
    assert len(polys) == 3
    assert all(isinstance(p, Polynomial) for p in polys)


def test_get_polynomial_weights_structure():
    layer = PolynomialLayer(input_dim=2, output_dim=2, max_degree=1)
    weights = layer.get_polynomial_weights()
    assert isinstance(weights, list) and len(weights) > 0
    flat = [p for group in weights for p in group]
    assert all(isinstance(p, Polynomial) for p in flat)


def test_early_stopping_patience():
    es = EarlyStopping(patience=2, min_delta=0.0)
    results = [es(v) for v in (1.0, 0.8, 0.9, 0.95)]
    assert results == [False, False, False, True]


def test_orthopoly_layer_additive_forward():
    layer = OrthoPolyLayer(input_dim=3, output_dim=2, max_degree=3,
                           rank=2, basis_type="chebyshev_T",
                           interaction_mode="additive")
    out = layer(torch.rand(4, 3) * 2.0 - 1.0)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_orthopoly_layer_multiplicative_forward():
    layer = OrthoPolyLayer(input_dim=3, output_dim=2, max_degree=3,
                           rank=2, basis_type="legendre",
                           interaction_mode="multiplicative")
    out = layer(torch.rand(4, 3) * 2.0 - 1.0)
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_orthopoly_layer_polynomial_weights_evaluable():
    layer = OrthoPolyLayer(input_dim=3, output_dim=1, max_degree=3,
                           rank=2, basis_type="chebyshev_T",
                           interaction_mode="additive")
    polys = layer.get_polynomial_weights()[0]
    assert len(polys) == 3
    for i, poly in enumerate(polys):
        val = float(poly.evaluate({Variable(f"x{i}"): 0.5}))
        assert np.isfinite(val)


def test_rational_layer_forward():
    layer = RationalPolyLayer(input_dim=2, output_dim=1, max_degree=3)
    out = layer(torch.randn(6, 2))
    assert out.shape == (6, 1)
    assert torch.isfinite(out).all()


def test_groebner_layer_projection_normal_form():
    gl = GroebnerLayer(input_dim=3, output_dim=4, constraint_ideal=["x0 - x1"])
    before = gl.weight_matrix.detach().clone().numpy().ravel()
    gl.project_weights()
    after = gl.weight_matrix.detach().numpy().ravel()

    # normal form modulo <x0 - x1>: leading coefficient killed,
    # w0 + w1 preserved
    assert after[0] == 0.0
    assert after[0] + after[1] == pytest.approx(before[0] + before[1], rel=1e-9)

    snapshot = gl.weight_matrix.detach().clone()
    gl.project_weights()
    assert torch.equal(snapshot, gl.weight_matrix)  # idempotent


def test_groebner_layer_membership_test():
    gl = GroebnerLayer(input_dim=3, output_dim=2, constraint_ideal=["x0 - x1"])
    assert gl.ideal_membership_test(np.zeros(6))          # 0 is in every ideal
    gen_vec = np.zeros(6); gen_vec[0], gen_vec[1] = 1.0, -1.0
    assert gl.ideal_membership_test(gen_vec)              # the generator itself
    outside = np.zeros(6); outside[0] = 1.0
    assert not gl.ideal_membership_test(outside)          # x0 not in <x0-x1>


def test_groebner_layer_forward_shape():
    gl = GroebnerLayer(input_dim=4, output_dim=2, constraint_ideal=[])
    out = gl(torch.randn(7, 4))
    assert out.shape == (7, 2)
    assert torch.isfinite(out).all()


def test_create_groebner_layer_factory():
    gl = create_groebner_layer(input_dim=3, output_dim=2,
                               constraint_ideal=["x0 - x1"])
    out = gl(torch.randn(3, 3))
    assert out.shape == (3, 2)
