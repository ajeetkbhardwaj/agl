import numpy as np
import pytest

from src.training.loss_functions import (
    CrossEntropy, HuberLoss, HingeLoss, IdealMembershipLoss,
    MeanAbsoluteError, MeanSquaredError, create_loss,
)
from src.training.optimizers import (
    Adam, PolynomialAwareOptimizer, SGD,
)


def test_mse_value_and_gradient():
    loss_fn = MeanSquaredError()
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[1.5], [2.5]])
    assert loss_fn(y_true, y_pred) == pytest.approx(0.25)
    grad = loss_fn.gradient(y_true, y_pred)
    assert grad.shape == y_pred.shape


def test_mae_value():
    loss_fn = MeanAbsoluteError()
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[1.5], [2.5]])
    assert loss_fn(y_true, y_pred) == pytest.approx(0.5)


def test_cross_entropy_matches_entropy_of_target():
    logits = np.array([[1.0, 2.0, 3.0]])
    e = np.exp(logits - logits.max())
    q = e / e.sum()
    ce = CrossEntropy(multi_class=True)
    value = ce(q, logits)
    entropy = float(-(q * np.log(q)).sum())
    assert value == pytest.approx(entropy, abs=1e-10)
    assert value == pytest.approx(0.832396, abs=5e-4)


def test_huber_quadratic_and_linear_regimes():
    huber = HuberLoss(delta=1.0)
    y_true = np.array([[0.0], [0.0]])
    y_pred = np.array([[0.5], [3.0]])
    # small residual: 0.5*r^2 ; large: delta*(|r| - delta/2)
    expected = (0.125 + 2.5) / 2
    assert huber(y_true, y_pred) == pytest.approx(expected)


def test_hinge_margin():
    hinge = HingeLoss(margin=1.0)
    y_true = np.array([[1.0], [-1.0]])
    y_pred = np.array([[0.5], [-0.8]])
    assert hinge(y_true, y_pred) == pytest.approx(0.35)


def test_ideal_membership_loss_callable_generator():
    gen = lambda yp: yp[:, 0] - yp[:, 1]
    iml = IdealMembershipLoss(ideal_generators=[gen])
    y_true = np.zeros((2, 2))
    on_variety = np.array([[1.0, 1.0], [2.0, 2.0]])
    off_variety = np.array([[1.0, 2.0], [3.0, 3.0]])

    base_on = float(np.mean(on_variety ** 2))          # MSE part only
    assert iml(y_true, on_variety) == pytest.approx(base_on)

    base_off = float(np.mean(off_variety ** 2))
    violation = float(np.mean((off_variety[:, 0] - off_variety[:, 1]) ** 2))
    assert iml(y_true, off_variety) == pytest.approx(base_off + violation)


def test_create_loss_factory():
    assert isinstance(create_loss("mse"), MeanSquaredError)
    assert isinstance(create_loss("huber", delta=0.5), HuberLoss)


class _DummyLayer:
    def __init__(self):
        self.weight_matrix = np.ones((2, 2))
        self.grad_weight = np.full((2, 2), 2.0)
        self.bias = None


class _DummyModel:
    def __init__(self):
        self.layers = [_DummyLayer()]


def test_sgd_step_updates_weights():
    model = _DummyModel()
    SGD(learning_rate=0.1).step(model)
    assert np.allclose(model.layers[0].weight_matrix, 0.8)


def test_adam_step_changes_weights_and_stays_bounded():
    model = _DummyModel()
    before = model.layers[0].weight_matrix.copy()
    Adam(learning_rate=0.01).step(model)
    after = model.layers[0].weight_matrix
    assert not np.allclose(before, after)
    assert np.all(np.abs(after) <= 10.0)


def test_polynomial_aware_optimizer_projection_idempotent():
    opt = PolynomialAwareOptimizer(learning_rate=0.01)
    W = np.array([[1.0], [0.2]])
    gen = lambda w: float(w.ravel()[0] - w.ravel()[1])
    projected = opt.project_to_variety(W, {"generators": [gen]})
    assert abs(gen(projected)) < 1e-5                      # constraint satisfied
    again = opt.project_to_variety(projected, {"generators": [gen]})
    assert np.allclose(projected, again)                   # idempotent
